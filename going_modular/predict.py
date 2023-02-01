import torch
import torchvision
import argparse
import model_builder

#creating a parser
parser = argparse.ArgumentParser()
#get an image path
parser.add_argument("--image",
                    help="target image to predict on")

#get a model path
parser.add_argument("--model_path",
                    default = "models/05_going_modular_script_mode_tinyvgg_model.pth",
                    type = str,
                    help = "target model to use for prediction  filepath")

args = parser.parse_args()

#setup class names
class_names = ["pizza", "steak", "sushi"]
#setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

#get the image path
IMG_PATH = args.image
print(f"[info] predictiin on {IMG_PATH}")

#fucntion to load in the model
def load_model(filepath = args.model_path):
  #need to use same hyperparamerter as saved model
  model = model_builder.TinyVGG(input_shape =3,
                                hidden_units = 128,
                                output_shape =3).to(device)
  print(f"[info] loading in model from : {filepath}")
  #load in the saved model state dictionary from file
  model.load_state_dict(torch.load(filepath))
  
  return model

#fuction to laod in model + predict on select image
def predict_on_image(image_path = IMG_PATH, filepath=args.model_path):
  #load the model
  model = load_model(filepath)
  #load in the image and turn it into torch.float32 (same type as model)
  image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

  #preprocess the image
  image = image /255.
  #resise as same size in the model
  transform = torchvision.transforms.Resize((64,64))
  image = transform(image)
  image = image.unsqueeze(dim=0)

  #predcit on image
  model.eval()
  with torch.inference_mode():
    image = image.to(device)
    pred_logits = model(image)
    pred_prob = torch.softmax(pred_logits, dim=1)
    pred_label = torch.argmax(pred_prob, dim=1)
    pred_label_class = class_names[pred_label]
  print(f"[info] pred class: {pred_label_class}, pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
  predict_on_image()
