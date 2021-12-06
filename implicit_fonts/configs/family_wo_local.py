# PS i know this is a bad way to write a config file since rogue code can be injected
# but I like the fexibility that comes with this
img_size = 128
max_sampling = 15000 # importance sampling parameter
batchsize=4
epochs = 3000

save_dir = '../logs/implicit_fonts/{}/{}'
steps_til_summary = 20
init_training =50
dataset_folder = './data/renders/'