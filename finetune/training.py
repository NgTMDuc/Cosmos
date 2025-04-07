import sys
sys.path.append("../")
from cosmos1.models.tokenizer.networks import TokenizerConfigs, TokenizerModels
import torch
from tqdm import tqdm
from dataset_loader import get_dataloader
from loguru import logger
from loss import SSIM3D
import numpy as np
import matplotlib.pyplot as plt
import os 

model_class = TokenizerModels.CV
config = TokenizerConfigs.CV8x8x8.value
logger.add("training_logs.log", rotation="100 MB", level="INFO", backtrace=True, diagnose=True)

def train_1_epoch(model, 
                  train_loader, 
                  val_loader, 
                  device,
                  optimizer, 
                  save_path,
                  logger,
                  loss_L1,
                  loss_SSIM = None
                  ):
    model.train()
    for batch in tqdm(train_loader):
        # Foward pass
        batch = batch.to(device)
        output = model(batch)

        l1 = loss_L1(output, batch)
        if loss_SSIM is not None:
            ssim = loss_SSIM(output, batch)
            loss = l1 + ssim
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info('train_loss', loss.item())
        logger.info('train_l1', l1.item())
        if loss_SSIM is not None:
            logger.info('train_ssim', ssim.item())
    
    model.eval()
    with torch.no_grad():
        i = 0
        for batch in tqdm(val_loader):
            i += 1
            batch = batch.to(device)
            output = model(batch)
            l1 = loss_L1(output, batch)
            if loss_SSIM is not None:
                ssim = loss_SSIM(output, batch)
                loss = l1 + ssim
            logger.info('val_loss', loss.item())
            logger.info('val_l1', l1.item())
            if loss_SSIM is not None:
                logger.info('val_ssim', ssim.item())
            
            if i % 100 == 0:
                save_combined_visualization(batch[0], output[0], save_path = "./images/{i}")
        
def save_combined_visualization(input_image, output_image, save_path, slice_step=10, num_slices=10):
    """
    Gộp tất cả các slice của ảnh input và output vào một bức ảnh lớn và lưu vào thư mục.

    Args:
    - input_image: Tensor 4D với shape (D, 3, H, W) của ảnh đầu vào.
    - output_image: Tensor 4D với shape (D, 3, H, W) của ảnh đầu ra.
    - save_path: Đường dẫn thư mục để lưu bức ảnh kết quả.
    - slice_step: Bước nhảy giữa các slice để chọn (mặc định là 10).
    - num_slices: Số lượng slice bạn muốn gộp và lưu (mặc định là 10).
    """
    # Lấy chiều D (số lượng slices) từ ảnh
    D = input_image.shape[0]  # Đảm bảo input_image có shape (D, 3, H, W)

    # Chọn các slice để hiển thị
    selected_slices = np.arange(0, D, slice_step)[:num_slices]

    # Gộp các slice lại thành một bức ảnh lớn
    combined_image = []

    for slice_idx in selected_slices:
        input_slice = input_image[slice_idx]  # Lấy slice từ ảnh đầu vào (shape: (3, H, W))
        output_slice = output_image[slice_idx]  # Lấy slice từ ảnh đầu ra (shape: (3, H, W))
        
        # Chuyển đổi các slice thành hình ảnh 3D (hợp nhất kênh)
        input_slice = np.moveaxis(input_slice.numpy(), 0, -1)  # Chuyển (3, H, W) thành (H, W, 3)
        output_slice = np.moveaxis(output_slice.numpy(), 0, -1)  # Chuyển (3, H, W) thành (H, W, 3)
        
        # Gộp input và output cạnh nhau
        combined_slice = np.concatenate([input_slice, output_slice], axis=1)
        combined_image.append(combined_slice)

    # Gộp tất cả các slice lại thành một ảnh lớn
    combined_image = np.concatenate(combined_image, axis=0)

    # Tạo thư mục nếu chưa có
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Lưu ảnh kết quả
    filename = os.path.join(save_path, "combined_slices.png")
    plt.imsave(filename, combined_image)

    print(f"Saved combined visualization to {filename}")

def training(cfg):
    model = torch.jit.load(cfg.pretrained_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = get_dataloader(cfg.train, mode = "train")
    val_loader = get_dataloader(cfg.val, mode = "val")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_L1 = torch.nn.L1Loss()
    if cfg.ssim:
        loss_SSIM = SSIM3D()
    
    save_path = cfg.save_path
    for epoch in range(cfg.EPOCH):
        logger.info(f"Epoch {epoch + 1}/{cfg.EPOCH} - Training Started")
        train_1_epoch(model, train_loader, val_loader, device, optimizer, save_path, logger, loss_L1, loss_SSIM)
        logger.info(f"Epoch {epoch + 1}/{cfg.EPOCH} - Training Finished")

    
if __name__ == "__main__":
    model = TokenizerModels.CV.value(**config)
    # with open("model_architecture.txt", "w") as f:
        # print(model, file=f)
    # model.load_state_dict(torch.load("/mnt/disk1/data/.ducntm/Cosmos/ckpt/Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit", weights_only=False))
    jit_model = torch.jit.load("/mnt/disk1/data/.ducntm/Cosmos/ckpt/Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit")
    state_dict = jit_model.state_dict()
    model.load_state_dict(state_dict)
    