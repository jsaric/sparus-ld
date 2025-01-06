import torch
from data.build_dataset import build_train_dataset, build_val_dataset
from evaluators.build import build_evaluators
from loss.keypoint_heatmap_loss import KeypointHeatmapLoss
from logger import Logger
from models.build_model import build_model
from config import get_cfg_defaults
import argparse
import time
import os
import tqdm


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.initialize_output_dir(cfg.OUTPUT_DIR)
        self.logger = Logger(cfg.OUTPUT_DIR, log_time=True)
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg)
        self.scheduler = self.build_scheduler(cfg)
        self.loss_fn = self.build_loss_fn(cfg)
        self.train_loader = self.build_train_loader(cfg)
        self.val_loader = self.build_val_loader(cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_iters = cfg.TRAINER.ITERATIONS
        self.output_dir = cfg.OUTPUT_DIR
        self.evaluators = self.build_evaluators(cfg, self.val_loader)
        self.ms_eval = cfg.TRAINER.MS_EVAL
        self.ms_scales = cfg.TRAINER.MS_SCALES

    def initialize_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            response = input("Output directory already exists. Do you want to overwrite it? (y/n)")
            if response != "y":
                exit(0)

        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(self.cfg.dump())

    def build_loss_fn(self, cfg):
        return KeypointHeatmapLoss(
            sigma=cfg.LOSS.SIGMA,
            top_k_percent_pixels=cfg.LOSS.TOP_K_PERCENT_PIXELS,
            criterion=cfg.LOSS.TYPE
        )

    def build_evaluators(self, cfg, val_loader):
        return build_evaluators(cfg, val_loader)

    def build_model(self, cfg):
        return build_model(cfg)

    def build_train_loader(self, cfg):
        dataset = build_train_dataset(cfg)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.TRAINER.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.TRAINER.NUM_WORKERS,
            pin_memory=True
        )

    def build_val_loader(self, cfg):
        dataset = build_val_dataset(cfg)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.TRAINER.NUM_WORKERS,
            pin_memory=True
        )

    def build_optimizer(self, cfg):
        if cfg.OPTIMIZER.TYPE == "adam":
            return torch.optim.Adam(
                [
                    {'params': self.model.train_params(), 'lr': cfg.OPTIMIZER.LR, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY},
                    {'params': self.model.fine_tune_params(), 'lr': cfg.OPTIMIZER.LR * cfg.OPTIMIZER.FINE_TUNE_LR_MULTIPLIER, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY}
                ],
            )
        else:
            raise NotImplementedError(f"Unknown optimizer: {cfg.OPTIMIZER.TYPE}")

    def save_checkpoint(self, iteration):
        torch.save(
            {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': iteration
            },
            os.path.join(self.output_dir, f"checkpoint_{iteration}.pth")
        )

    def build_scheduler(self, cfg):
        if cfg.OPTIMIZER.SCHEDULER.TYPE == "PolyLR":
            return torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                cfg.OPTIMIZER.SCHEDULER.POLYLR.MAX_ITER,
                cfg.OPTIMIZER.SCHEDULER.POLYLR.POWER
            )
        elif cfg.OPTIMIZER.SCHEDULER.TYPE == "ExpLR":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                cfg.OPTIMIZER.SCHEDULER.EXPLR.GAMMA
            )
        else:
            raise NotImplementedError(f"Unknown learning rate scheduler: {cfg.OPTIMIZER.SCHEDULER.TYPE}")

    def train(self):
        generator = iter(self.train_loader)
        for i in range(self.max_iters):
            try:
                data = next(generator)
            except StopIteration:
                generator = iter(self.train_loader)
                data = next(generator)

            images = data['image'].to(self.device)
            out = self.model(images)
            loss = self.loss_fn(out["heatmap"], data["keypoints"].to(self.device))
            loss.backward()
            self.optimizer.step()
            if i % self.cfg.OPTIMIZER.SCHEDULER.STEP_INTERVAL == 0:
                self.scheduler.step()
            self.optimizer.zero_grad()

            if i % self.cfg.TRAINER.LOG_INTERVAL == 0:
                self.logger.log(f"Iter: {i}, loss: {loss.item()}, learning_rate: {self.optimizer.param_groups[0]['lr']}")
            if i % self.cfg.TRAINER.EVAL_INTERVAL == 0:
                self.logger.log(f"Evaluating at iter: {i}")
                results = self.evaluate()
                self.logger.log("Evaluation results: ")
                self.logger.log(results)
            if i % self.cfg.TRAINER.CHECKPOINT_INTERVAL == 0:
                self.logger.log(f"Saving checkpoint at iter: {i}")
                self.save_checkpoint(i)

        self.save_checkpoint(self.max_iters)

    def evaluate(self):
        self.evaluators.reset()
        self.model.eval()
        for i, data in tqdm.tqdm(enumerate(self.val_loader)):
            images = data['image'].to(self.device)
            with torch.no_grad():
                if self.ms_eval:
                    out = self.model.forward_ms(images, self.ms_scales)
                else:
                    out = self.model(images)
            self.evaluators.update(out, data)
        results = self.evaluators.evaluate()
        self.model.train()
        return results


parser = argparse.ArgumentParser()
parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
parser.add_argument("--eval-only", action="store_true", help="perform evaluation only", default=False)
parser.add_argument("--model-weights", default="", metavar="FILE", help="path to model weights", type=str)
parser.add_argument(
    "opts",
    help=""" Modify config options at the end of the command. For Yacs configs, use space-separated "PATH.KEY VALUE" pairs.""".strip(),
    default=None,
    nargs=argparse.REMAINDER,
)

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    trainer = Trainer(cfg)
    if args.model_weights:
        trainer.logger.log("Loading model weights from: " + args.model_weights)
        trainer.model.load_state_dict(torch.load(args.model_weights)["model"])
    if args.eval_only:
        trainer.logger.log("Performing evaluation only...")
        trainer.logger.log(trainer.evaluate())
    else:
        trainer.train()