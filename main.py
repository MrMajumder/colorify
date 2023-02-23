import argparse
from layers.archiRunner import Training

# hyperparameters
BATCHSIZE = 32
EPOCH = 45
LEARNINGRATE = 0.001
# utility parameters
num_workers = 12
img_dir_train = "./data/places10/train"
img_dir_val = "./data/places10/val"
img_dir_test = "./data/places10/test"
img_out_dir = "./out/places10"
models_dir = "./model/places10"


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Colorify: B&W to colorful images",formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', help='train: to train, test: to test, none: train+test')
    parser.add_argument('--model', help='Path to pretrained .pt model')
    args = parser.parse_args()
    
    trainer = Training(batch_size=BATCHSIZE,
                         epochs=EPOCH,
                         learning_rate=LEARNINGRATE,
                         img_dir_train=img_dir_train,
                         img_dir_val=img_dir_val,
                         img_dir_test=img_dir_test,
                         model_checkpoint=args.model,
                         num_workers=num_workers,
                         models_dir=models_dir,
                         img_out_dir=img_out_dir,
                         )
    trainer.info()
    # input("Stuff loaded and working")
    if(args.mode is None):
        trainer.run()
        trainer.test()
    elif(args.mode == "train"):
        trainer.run()
    else:
        trainer.test()
