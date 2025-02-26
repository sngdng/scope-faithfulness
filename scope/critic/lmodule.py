import lightning as L
import torch
import torchmetrics


class ClassificationModule(L.LightningModule):
    """
    Train the models with regression loss (MSE by default)
    """

    def __init__(self, model, loss=torch.nn.BCEWithLogitsLoss(), lr=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_func = loss
        self.lr = lr if lr is not None else 5e-5
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.val_prec = torchmetrics.Precision(task="binary")
        self.test_prec = torchmetrics.Precision(task="binary")
        self.val_rec = torchmetrics.Recall(task="binary")
        self.test_rec = torchmetrics.Recall(task="binary")
        self.pretrained_model_fields = [
            "input_ids",
            "attention_mask",
        ]  # 'token_type_ids',

    def _forward(self, batch):
        model_input = {k: batch[k] for k in self.pretrained_model_fields}
        output = self.model(**model_input)
        return torch.squeeze(output)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._forward(batch)

    def training_step(self, batch, batch_idx):
        outputs = self._forward(batch)
        loss = self.loss_func(outputs, batch["labels"].float())
        self.train_acc(outputs, batch["labels"])
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._forward(batch)
        loss = self.loss_func(outputs, batch["labels"].float())
        self.val_acc(outputs, batch["labels"])
        self.val_f1(outputs, batch["labels"])
        self.val_prec(outputs, batch["labels"])
        self.val_rec(outputs, batch["labels"])
        self.log("val_loss", loss)
        self.log("valid_acc", self.val_acc, on_step=True, on_epoch=True)
        self.log("valid_f1", self.val_f1, on_step=True, on_epoch=True)
        self.log("valid_prec", self.val_prec, on_step=True, on_epoch=True)
        self.log("valid_rec", self.val_rec, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self._forward(batch)
        test_loss = self.loss_func(outputs, batch["labels"].float())
        self.log("test_loss", test_loss)
        self.test_acc(outputs, batch["labels"])
        self.test_f1(outputs, batch["labels"])
        self.test_prec(outputs, batch["labels"])
        self.test_rec(outputs, batch["labels"])
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_prec", self.test_prec, on_step=False, on_epoch=True)
        self.log("test_rec", self.test_rec, on_step=False, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # optimizer = bnb.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer  # better results without scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
        # return [optimizer], [scheduler]


class SoftmaxClassificationModule(ClassificationModule):
    """
    Train the models with regression loss (MSE by default)
    """

    def __init__(self, model, loss=torch.nn.CrossEntropyLoss(), lr=None):
        super().__init__(model, loss, lr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.pretrained_model_fields = ["input_ids", "attention_mask", "labels"]

    def _forward(self, batch):
        model_input = {k: batch[k] for k in self.pretrained_model_fields}
        output = self.model(**model_input)
        return output.logits, output.loss

    def training_step(self, batch, batch_idx):
        outputs, loss = self._forward(batch)
        # loss = self.loss_func(outputs, batch["labels"])
        self.train_acc(outputs.max(1).indices, batch["labels"])
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, _ = self._forward(batch)
        y_pred = outputs.max(1).indices
        loss = self.loss_func(outputs, batch["labels"])
        self.val_acc(y_pred, batch["labels"])
        self.val_f1(y_pred, batch["labels"])
        self.val_prec(y_pred, batch["labels"])
        self.val_rec(y_pred, batch["labels"])
        self.log("val_loss", loss)
        self.log("valid_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("valid_f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("valid_prec", self.val_prec, on_step=False, on_epoch=True)
        self.log("valid_rec", self.val_rec, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs, _ = self._forward(batch)
        y_pred = outputs.max(1).indices
        test_loss = self.loss_func(outputs, batch["labels"])
        self.log("test_loss", test_loss)
        self.test_acc(y_pred, batch["labels"])
        self.test_f1(y_pred, batch["labels"])
        self.test_prec(y_pred, batch["labels"])
        self.test_rec(y_pred, batch["labels"])
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_prec", self.test_prec, on_step=False, on_epoch=True)
        self.log("test_rec", self.test_rec, on_step=False, on_epoch=True)
        return test_loss
