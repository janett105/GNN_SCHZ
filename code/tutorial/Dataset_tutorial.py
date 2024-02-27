
import torch
from torch_geometric.data import Data

# 하나의 edge에 대해 양방향으로 2개 index쌍을 써야 함
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# edge_index를 list of index tuple로 만들고(.t()), 변환 후에 메모리 연속적으로 만듦(.contiguous())
data = Data(x=x, edge_index=edge_index.t().contiguous())

"""
Graph 데이터 변환 : torch_geometric.transforms
pre_transform : disk memory 할당 전에 변환
T.KnnGraph(k=) : KNN을 통해 그래프 형태로 변형
T.RandomTranslate() : 각 node의 위치를 조금씩 이동(perturbation)
"""
# from torch_geometric.datasets import ShapeNet
# import torch_geometric.transforms as T

# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
# # 3차원 점 데이터, edge_index가 없음(그래프 X)
# print(f'변환 전 : {dataset[0]}')

# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
#                    pre_transform=T.KNNGraph(k=6),
#                    transform=T.RandomTranslate(0.01))
# print(f'변환 후 : {dataset[0]}')
"""
creating GNN dataset : torch_geometric.data.InMemoryDataset
torch_geometric.data.Dataset 상속받음
whole dataset이 CPU memory 사용 가능할 때 사용

<dataset 저장 root folder>
1. raw_dir(처리되어야 하는 것)
2. processing_dir(처리된 것)

<dataset function (default : None)>
1. transform : data accessing 전 동적으로 변환, data augmentation에서 사용
2. pre_transform : disk 저장 전에 변환, 한 번만 하면 되는 cost 높은 변환에서 사용
3. pre_filter : data object 저장 전에 filter

1. raw_file_names() : 다운 건너뛸 raw_dir에서의 file 목록
    raw_paths : 해당 file들의 dir 
2. processed_file_names() : 처리 과정을 건너 뛸 process_dir에서의 file 목록
    processed_paths : 해당 file들의 dir
3. download() : raw_dir에서 raw data download
4. process() : raw data processing, processing_dir에 save 
"""
import torch
from torch_geometric.data import InMemoryDataset, download_url

class CustomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0]) # data, slices load
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        return download_url(url, self.raw_dir)
    
    def process(self):
        data_list=[]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])