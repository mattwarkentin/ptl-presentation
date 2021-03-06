import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks.progress import RichProgressBar
import torchmetrics as tm
import monai

def main():
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
            self.mnist_train, self.mnist_val = random_split(mnist, [48000, 12000])
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=8)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=8)

    class MySweetModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = monai.networks.nets.ViT(
                in_channels=1,
                img_size=(28,28),
                patch_size=(7,7),
                hidden_size=128,
                mlp_dim=256,
                num_layers=4,
                num_heads=4,
                spatial_dims=2,
                classification=True,
                num_classes=10
            )
            self.loss = torch.nn.CrossEntropyLoss()
            self.acc_train = tm.Accuracy(num_classes=10)
            self.acc_valid = tm.Accuracy(num_classes=10)
            self.acc_test = tm.Accuracy(num_classes=10)
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        patience=5,
                        verbose=True
                    ),
                    "monitor": "valid_loss"
                }
            }

        def forward(self, x):
            return self.model(x)[0]

        def training_step(self, batch, batch_id):
            x,y_true = batch
            y_pred = self(x)
            loss = self.loss(y_pred, y_true)
            self.log('train_loss', loss, prog_bar=True)
            y_prob = y_pred.softmax(dim=-1)
            self.acc_train(y_prob, y_true)
            self.log('train_acc', self.acc_train, on_step=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_id):
            x,y_true = batch
            y_pred = self(x)
            loss = self.loss(y_pred, y_true)
            self.log('valid_loss', loss, prog_bar=True)
            y_prob = y_pred.softmax(dim=-1)
            self.acc_valid(y_prob, y_true)
            self.log('valid_acc', self.acc_valid, on_epoch=True, on_step=False)
            return loss

        def test_step(self, batch, batch_id):
            x,y_true = batch
            y_pred = self(x)
            y_prob = y_pred.softmax(dim=-1)
            self.acc_valid(y_prob, y_true)
            self.log('test_acc', self.acc_valid, on_step=False, on_epoch=True)
            return None

    log_csv = CSVLogger('example/lightning_logs', 'metrics')
    log_tb = TensorBoardLogger('example/lightning_logs', 'tensorboard')
    loggers = [log_csv, log_tb]

    cb_progress = RichProgressBar()
    cb_earlystop = EarlyStopping(
        monitor='valid_loss',
        mode='min',
        patience=10,
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
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        limit_test_batches=0.1,
        callbacks=callbacks,
        logger=loggers,
        accelerator='auto',
        devices='auto',
        #profiler='simple',
        #fast_dev_run=True,
        #val_check_interval=0.25,
        #overfit_batches=25,
        #log_every_n_steps=5
    )

    datamodule = MySweetData()
    model = MySweetModel()

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == '__main__':
    main()
