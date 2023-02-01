import os
import argparse
import torch 
from torchvision import transforms
import data_setup, engine, model_builder, utils

#create a parser
parser = argparse.ArgumentParser(description = "get some hyperparameters.")

#get an arg for num_epochs
parser.add_argument("--num_epochs",
                    default =10,
                    type=int,
                    help="the number of epochs to train for")
#get an arg for batch_size
parser.add_argument("--batch_size",
                    default = 32,
                    type = int,
                    help = "the number of sample per batch ")
#get an arg for hidden_units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help = " number of hidden units in hidden layers")
#get an arg for learning rate
parser.add_argument("--learning_rate",
                    default = 0.001,
                    type = float,
                    help= "learning rate to use for model")

#create an arg for training directory
parser.add_argument("--train_dir",
                    default = "data/pizza_steak_sushi/train",
                    type = str,
                    help = "directory file path to training data in standard image classification format")

#create an arg for test directory
parser.add_argument("--test_dir",
                    default = "data/pizza_steak_sushi/test",
                    type = str,
                    help = "directory file path to test data in standard image classificat")
#get out arguments from the parser
args = parser.parse_args()
#setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS =args.hidden_units
LEARNING_RATE = args.learning_rate
print(f"[info] training a model for {NUM_EPOCHS} epochs with a {BATCH_SIZE} batch_size using {HIDDEN_UNITS} hidden unist and a learning rate {LEARNING_RATE} ")
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[info] training data file: {train_dir}")
print(f"[info] testing data file: {test_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"
data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = train_dir, 
    test_dir = test_dir,
    transform = data_transform,
    batch_size = BATCH_SIZE
)
model = model_builder.TinyVGG(
    input_shape =3,
    hidden_units = HIDDEN_UNITS,
    output_shape = len(class_names)
).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
engine.train(model = model,
             train_dataloader = train_dataloader,
             test_dataloader = test_dataloader,
             loss_fn = loss_fn,
             optimizer = optimizer,
             epochs = NUM_EPOCHS,
             device = device)
utils.save_model(model=model,
                 target_dir = "models",
                 model_name = "05_going_modular_script_mode_tinyvgg_model.pth")
