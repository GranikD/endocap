from PIL import Image

def predict(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        class_id = output.argmax(1).item()

    return dataset.classes[class_id]

print(predict("test_frame.jpg"))
