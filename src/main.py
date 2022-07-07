import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)    # 让每次随机初始化相同
checkpoint = utility.checkpoint(args)


def main():
    global model    # 这是在main函数中使用model所以声明是全局变量，并对全局model进行了赋值
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            # 变量
            # 变量前带_表明是一个私有变量，只用于标明，外部类还是可以访问到这个变量
            # 变量前带两个_,后带两个_：表明是内置变量
            # 大写加下划线的变量：标明是不会发生改变的全局变量
            # 函数
            # 前带_，标明是一个私有函数，只用于标明
            # 前带两个_，后带两个_的函数：标明是特殊函数
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
