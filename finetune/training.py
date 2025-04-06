import sys
sys.path.append("../")
from cosmos1.models.tokenizer.networks import TokenizerConfigs, TokenizerModels
import torch
from tqdm import tqdm

model_class = TokenizerModels.CV
config = TokenizerConfigs.CV8x8x8.value
print(config)
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
        logger.add_scalar('train_loss', loss.item())
        logger.add_scalar('train_l1', l1.item())
        if loss_SSIM is not None:
            logger.add_scalar('train_ssim', ssim.item())
    
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
            logger.add_scalar('val_loss', loss.item())
            logger.add_scalar('val_l1', l1.item())
            if loss_SSIM is not None:
                logger.add_scalar('val_ssim', ssim.item())
            
            if i % 100 == 0:
                save_visualization(batch[0], output[0])
        
    # Save the visualization
def save_visualization(image, output):
    # save the visualization
    pass

def training(cfg):
    model = torch.jit.load(cfg.pretrained_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = load_dataset(cfg.train, mode = "train")
    val_loader = load_dataset(cfg.val, mode = "val")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_L1 = torch.nn.L1Loss()
    if cfg.ssim:
        loss_SSIM = torch.nn.SSIM(window_size=11, reduction='mean')
    
    save_path = cfg.save_path
    
if __name__ == "__main__":
    model = TokenizerModels.CV.value(**config)
    # with open("model_architecture.txt", "w") as f:
        # print(model, file=f)
    # model.load_state_dict(torch.load("/mnt/disk1/data/.ducntm/Cosmos/ckpt/Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit", weights_only=False))
    jit_model = torch.jit.load("/mnt/disk1/data/.ducntm/Cosmos/ckpt/Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit")
    state_dict = jit_model.state_dict()
    model.load_state_dict(state_dict)
    