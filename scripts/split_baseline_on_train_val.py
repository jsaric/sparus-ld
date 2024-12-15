from tps.tps_io import TPSFile, TPSImage, TPSPoints


def is_val(img):
    val_prefixes = ["divlje_komarce_wv", "uzgoj_ita_organic_br.", "uzgoj_ita_organic_1_br.", "tunakavez_kali_wk"]
    return any([img.lower().startswith(p) for p in val_prefixes])


def split_baseline_tps_on_train_val(baseline_tps):
    train_images = []
    val_images = []
    for image in baseline_tps.images:
        if is_val(image.image):
            val_images.append(image)
        else:
            train_images.append(image)
    return TPSFile(train_images), TPSFile(val_images)
