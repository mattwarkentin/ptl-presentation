import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks.progress import RichProgressBar
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision as tv

class MySweetData(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.data_dir = 'example/mnist'
        self.batch_size = batch_size
    
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False)

class MySweetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(10816, 10)
        )
        self.loss = torch.nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        x,y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_id):
        x,y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss, prog_bar=True)
        return loss

datamodule = MySweetData()
model = MySweetModel()

log_csv = CSVLogger('example/lightning_logs', 'metrics')
log_tb = TensorBoardLogger('example/lightning_logs', 'tensorboard')
loggers = [log_csv, log_tb]

cb_progress = RichProgressBar()
cb_earlystop = EarlyStopping(
    monitor='valid_loss',
    mode='min',
    patience=5,
    check_on_train_epoch_end=False,
    verbose=True
)
cb_chkpt = ModelCheckpoint(
    dirpath=f'example/lightning_logs/checkpoints/',
    monitor='valid_loss',
    mode='min',
    save_top_k=1,
    filename='{epoch}_{step}_{valid_loss:.3f}'
)
callbacks = [cb_progress, cb_earlystop, cb_chkpt]

trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=10,
    callbacks=callbacks,
    logger=loggers,
    profiler='simple'
)

trainer.fit(model, datamodule)
