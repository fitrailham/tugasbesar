from Preparation import load_dataset, train

x, y = load_dataset('train')
x_val, y_val = load_dataset('validation')

# proses training, parameter, data augmentasi dan penyimpanan model ada di dalam fungsi train
# default lr = 1e-3, epoch = 10, data augmentasi horizontal flip, save model = False
train(x, y, x_val=x_val, y_val=y_val, save_model=True)