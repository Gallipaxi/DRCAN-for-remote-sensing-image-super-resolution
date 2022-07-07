from importlib import import_module
# import_module一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:  # set this option to test the model,default=''
            datasets = []
            for d in args.data_train:   # train dataset name, default='DIV2K'
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'   # module_name = 'DIV2K'
                m = import_module('data.' + module_name.lower())    # m = data.div2k
                datasets.append(getattr(m, module_name)(args, name=d))  # getattr(data.div2k, DIV2K)(args, name = DIV2K)

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:    # test dataset name， default='DIV2K'
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'   # module_name='DIV2K'
                m = import_module('data.' + module_name.lower())    # m=data.div2k()
                testset = getattr(m, module_name)(args, train=False, name=d)    # testset=data.div2k.DIV2K(args, train, name)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
