#coding:utf8
import warnings
class DefaultConfig(object):

    train_data_root = '/data2/yangyanwu_workplace/brain_age_2mm_all.h5'
    load_model_path = './checkpoints/brain_age_epoch_193' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 4 #96#28 # batch size 50-24 10-128
    use_gpu = True # user GPU or not
    pretrain = True
    num_workers = 0 # how many workers for loading data
    print_freq = int(4786 / batch_size) - 1#int(4786 / batch_size)  # print info every N batch

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
    
    max_epoch = 500
    lr = 0.001 # initial learning rate 5
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5 # 损失函数
    env = "main"

    gpu_num = 7

    nodes = 1
    gpus = 1
    nr = 0
    world_size = 1


def parse(self,kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))


DefaultConfig.parse = parse
opt =DefaultConfig()
# opt.parse = parse
