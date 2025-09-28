import os
from ultralytics import YOLO
import cv2

def predict_source(source, model_path=r"C:\Users\gxldx\OneDrive\Desktop\Computer Science\Programs\Python\Computer Vision\Dronolab Reconaissance Logo\models\balloon_exp\weights\best.pt", conf=0.5, show=True):
    """
    Perform YOLO predictions on the specified source (image, video, or webcam).

    Args:
        source (str or int): Path to an image or video file, or 0 for webcam.
        model_path (str): Path to the YOLO model file.
        conf (float): Confidence threshold for predictions.
        show (bool): Whether to display the predictions.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Check if the source is a webcam (integer 0 or other)
    if isinstance(source, int) or source == "webcam":
        cap = cv2.VideoCapture(0 if source == "webcam" else source)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Run inference on each frame
            model.predict(source=frame, conf=conf, show=show)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # For image or video files
        model.predict(source=source, conf=conf, save=True, show=show)
        print(f"Predictions completed for: {source}")

def _resolve_data_yaml_path(dataset_path):
    dataset_path = dataset_path.strip().strip('""').strip("'")
    if os.path.isfile(dataset_path) and dataset_path.lower().endswith(('.yaml', '.yml')):
        return os.path.abspath(dataset_path)
    
    if os.path.isdir(dataset_path):
        for name in ["data.yaml", "data.yml"]:
            potential_path = os.path.join(dataset_path, name)
            if os.path.isfile(potential_path):
                return os.path.abspath(potential_path)
    
    raise FileNotFoundError(f"No data.yaml or data.yml found in {dataset_path}")


def train_model(
    dataset_path: str,
    base_model: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = -1,
    device: str | None = None,
    project: str = "models",
    name: str | None = None,
):
    """
    Notes:

    base_model: Le model de base à utiliser pour entraîner avec Yolo11 (Pour l'instant on utilise nano)
    epochs: Nombre réeitiration
    imgsz: taille de l'image
    device: gpu ou cpu

    """
    data_yaml = _resolve_data_yaml_path(dataset_path)

    model = YOLO(base_model)

    print("\n=== Training config ===")
    print(f"data:    {data_yaml}")
    print(f"model:   {base_model}")
    print(f"epochs:  {epochs}")
    print(f"imgsz:   {imgsz}")
    print(f"batch:   {batch}")
    print(f"device:  {device if device is not None else 'auto'}")
    print(f"project: {project}")
    if name:
        print(f"name:    {name}")
    print("=======================\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device is not None else 0,
        project=project,
        name=name,
        verbose=True,
    )

    run_dir = getattr(results, "save_dir", None)
    if run_dir is None:
        try:
            subdirs = [os.path.join(project, d) for d in os.listdir(project)]
            subdirs = [d for d in subdirs if os.path.isdir(d)]
            run_dir = max(subdirs, key=os.path.getmtime)
        except Exception:
            run_dir = None

    if run_dir:
        best_weights = os.path.join(run_dir, "weights", "best.pt")
        print(f"\nTraining finished. Best weights :\n  {best_weights}")
        if os.path.isfile(best_weights):
            print("\nYou can run inference with:")
            print(f"predict_source(source='your_image_or_video', model_path='{best_weights}', conf=0.5)")
    else:
        print("\nTraining finished.")


if __name__ == "__main__":
    print("Select an input type:")
    print("1. Webcam")
    print("2. Image")
    print("3. Video")
    print("4. Train model with dataset")
    choice = input("Enter choice (1/2/3/4): ").strip()

    if choice == "1":
        predict_source(source="webcam")
    elif choice == "2":
        image_path = input("Enter the path to the image: ").strip()
        predict_source(source=image_path)
    elif choice == "3":
        video_path = input("Enter the path to the video: ").strip()
        predict_source(source=video_path)
    elif choice == "4":
        dataset_path = input("Enter the path to the dataset folder or data.yaml: ").strip()
        try:
            epochs = int(input("Epochs (default 50): ").strip() or "50")
            imgsz = int(input("Image size (default 640): ").strip() or "640")
            batch = int(input("Batch size (-1 auto): ").strip() or "-1")
        except ValueError:
            print("Invalid numeric input. Using defaults: epochs=50, imgsz=640, batch=-1")
            epochs, imgsz, batch = 50, 640, -1

        base_model = input("Base model (default yolo11n.pt): ").strip() or "yolo11n.pt"
        device = input("Device [cpu/mps/cuda/0/0,1 or empty=auto]: ").strip() or None

        train_model(
            dataset_path=dataset_path,
            base_model=base_model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device if device not in ("", "none", "auto") else None,
            project="models",
            name="balloon_exp",
        )

    else:
        print("Invalid choice. Please run the script again.")
