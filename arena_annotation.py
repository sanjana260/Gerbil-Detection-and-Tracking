import cv2
import numpy as np

class DraggableRectangle:
    def __init__(self, frame, points):

        self.original = frame
        self.image = frame.copy()

        self.points = points.astype(int)

        self.selected_point = None
        self.radius = 15

        cv2.namedWindow("Adjust Rectangle")
        cv2.setMouseCallback(
            "Adjust Rectangle",
            self.mouse_callback
        )

    def draw(self):

        self.image = self.original.copy()

        # Draw rectangle lines
        for i in range(4):
            pt1 = tuple(self.points[i])
            pt2 = tuple(self.points[(i+1) % 4])

            cv2.line(
                self.image,
                pt1,
                pt2,
                (0,255,0),
                2
            )

        # Draw corner points
        for pt in self.points:
            cv2.circle(
                self.image,
                tuple(pt),
                6,
                (0,0,255),
                -1
            )

    def get_closest_point(self, x, y):

        for i, pt in enumerate(self.points):

            dist = np.linalg.norm(
                np.array([x,y]) - pt
            )

            if dist < self.radius:
                return i

        return None

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            idx = self.get_closest_point(x, y)

            if idx is not None:
                self.selected_point = idx

        elif event == cv2.EVENT_MOUSEMOVE:

            if self.selected_point is not None:

                self.points[self.selected_point] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:

            self.selected_point = None

    def run(self):

        print("\nInstructions:")
        print("Drag red points to adjust rectangle")
        print("Press 'c' to confirm")
        print("Press 'r' to reset")

        initial_points = self.points.copy()

        while True:

            self.draw()

            cv2.imshow(
                "Adjust Rectangle",
                self.image
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                break

            elif key == ord("r"):
                self.points = initial_points.copy()

        cv2.destroyAllWindows()

        return self.points


def adjust_rectangle(frame, initial_points):

    tool = DraggableRectangle(
        frame,
        initial_points
    )

    return tool.run()