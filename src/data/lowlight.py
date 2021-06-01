import os
from data import srdata
import glob

class LowLight(srdata.SRData):
    def __init__(self, args, name='LowLight', train=True, benchmark=False):
        super(LowLight, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.args = args

    def _set_filesystem(self, dir_data):
        super(LowLight, self)._set_filesystem(dir_data)
        self.dir_hr = self.args.dir_hr
        self.dir_lr = self.args.dir_lr
        
        self.ext = (self.args.ext_hr, self.args.ext_lr) 

    def _scan(self):
        names_hr, names_lr = super(LowLight, self)._scan()

        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = names_lr[self.begin - 1:self.end]

        return names_hr, names_lr
