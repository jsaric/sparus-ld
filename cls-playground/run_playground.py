import pickle
import tkinter as tk
import numpy as np


class KeypointAdjuster:
    def __init__(self, root, model, width=768, height=512):
        self.root = root
        self.canvas = tk.Canvas(root, width=width, height=height, bg="white")
        self.canvas.pack()

        self.keypoints = default_keypoints.copy()
        self.circles = []
        self.texts = []
        self.scale_factor = 30  # Scale factor to map canvas coordinates to keypoint range (-20 to 20)
        self.classification_text = None
        self.offset_x = 350
        self.offset_y = 300
        self.width = width
        self.height = height
        self.model = model

        # Draw initial keypoints
        for i, (x, y) in enumerate(self.keypoints):
            canvas_x = self.offset_x + x * self.scale_factor
            canvas_y = self.offset_y - y * self.scale_factor
            circle = self.canvas.create_oval(canvas_x - c, canvas_y - c, canvas_x + c, canvas_y + c, fill="red",
                                             tags=f"kp{i}")
            self.circles.append(circle)
            self.canvas.tag_bind(f"kp{i}", "<B1-Motion>", self.move_keypoint)

            # Add text label for the keypoint
            text = self.canvas.create_text(canvas_x + 10, canvas_y, text=str(i + 1), fill="black", font=("Arial", 10))
            self.texts.append(text)

        # Add a classify button
        self.button = tk.Button(root, text="Classify", command=self.classify_keypoints)
        self.button.pack()

    def move_keypoint(self, event):
        # Find which keypoint is being dragged and update its position
        for i, circle in enumerate(self.circles):
            if self.canvas.type(circle) == "oval":
                coords = self.canvas.coords(circle)
                if coords[0] <= event.x <= coords[2] and coords[1] <= event.y <= coords[3]:
                    self.canvas.coords(circle, event.x - c, event.y - c, event.x + c, event.y + c)
                    # Update keypoints to reflect new canvas position
                    self.keypoints[i] = [(event.x - self.offset_x) / self.scale_factor, (self.offset_y - event.y) / self.scale_factor]

                    # Update text label position
                    self.canvas.coords(self.texts[i], event.x + 10, event.y)
                    self.classify_keypoints()

    def classify_keypoints(self):
        # Flatten keypoints for classification
        keypoints_array = self.keypoints.flatten()

        # Predict class using the trained MLP
        predicted_class = self.model.predict([keypoints_array])[0]
        predicted_prob = self.model.predict_proba([keypoints_array])[0]

        # Display the classification result
        if self.classification_text is not None:
            self.canvas.delete(self.classification_text)
        self.classification_text = self.canvas.create_text(self.width / 2, 20, text=f"Predicted Class: {classes[predicted_class]} P={max(predicted_prob):.2f}", fill="red", font=("Arial", 16))

if __name__ == '__main__':
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Step 2: Define default keypoints and their initial positions
    default_keypoints = np.loadtxt("default_keypoint.txt")

    c = 3

    classes = {
        0: "Wild",
        1: "Farmed"
    }

    # Create the tkinter application
    root = tk.Tk()
    root.title("Keypoint Adjuster and Classifier")
    app = KeypointAdjuster(root, model)
    root.mainloop()
