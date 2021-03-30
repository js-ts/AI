
import paddle
import torch

class Dataset(paddle.io.Dataset):
    def __init__(self, ):
        self.data = paddle.arange(0, 24).reshape((6, 4))
        print(self.data.shape)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self, ):
        return len(self.data)


class DatasetT(torch.utils.data.Dataset):

    def __init__(self, ):
        self.data = torch.arange(0, 24).reshape(2, 3, 4)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':

    dataset = Dataset()
    dataloader = paddle.io.DataLoader(dataset, batch_size=3)

    for batch in dataloader:
        print(batch)
        print(type(batch), len(batch))

    # datasett = DatasetT()
    # dataloadert = torch.utils.data.DataLoader(datasett, batch_size=3)

    # for batch in dataloadert:
    #     print(batch)
