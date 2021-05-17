
import argparse

class _Options(object):
    def __init__(self) -> None:
        super().__init__()
        self.parser = argparse.ArgumentParser('')
        self.add_options(self.parser)

    def parse(self, ):
        return self.parser.parse_args()

    def add_options(self, parser):
        raise NotImplementedError


class OptionsV1(_Options):
    def add_options(self, parser):
        
        parser.add_argument('--name', type=str, default='resnet18')

