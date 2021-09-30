import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models.gcn
import torch
from utils.losses import binary_cross_entropy
from utils.metric import accuracy
import os
import utils.data.functions as data_fuc
# import tasks
# import utils.callbacks
import utils.data
# import utils.email
# import utils.logging


# DATA_PATHS = {
#     "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
#     "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
# }

DATA_PATHS = {
    "train": "data/train",
    "test": "data/test",
    "adj": "data/plv_adj.csv"
}

def get_adj():
    return torch.Tensor(utils.data.functions.load_adjacency_matrix(DATA_PATHS["adj"]))

def get_feature_dataset(args):
    '''
    :return: torch-dataset: train_dataset, test_dataset
    '''
    return utils.data.functions.generate_torch_datasets(
        DATA_PATHS['train'],
        DATA_PATHS['test'],
        args.seq_len,
        args.normalize
    )
def get_model(args,adj):
    model = None
    if args.model_name == "GCN":
        model = models.gcn.GCNModel(num_layer=args.num_layer, adj=adj, input_dim=args.seq_len,
                                    hidden_dim=args.hidden_dim, output_dim=args.output_dim)
    # if args.model_name == "GRU":
    #     model = models.GRU(input_dim=adj.shape[0], hidden_dim=args.hidden_dim)
    # if args.model_name == "TGCN":
    #     model = models.TGCN(adj=adj, hidden_dim=args.hidden_dim)
    return model

# def get_model(args, dm):
#     model = None
#     if args.model_name == "GCN":
#         model = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
#     if args.model_name == "GRU":
#         model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
#     if args.model_name == "TGCN":
#         model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
#     return model


# def get_task(args, model, dm):
#     task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
#         model=model, feat_max_val=dm.feat_max_val, **vars(args)
#     )
#     return task


# def get_callbacks(args):
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
#     plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
#     callbacks = [
#         checkpoint_callback,
#         plot_validation_predictions_callback,
#     ]
#     return callbacks


def train(
        net,
        train_dataset,
        test_dataset,
        num_epoch,
        learning_rate,
        weight_decay,
        batch_size
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    train_feature, train_label = train_dataset[:]
    test_feature, test_label = test_dataset[:]

    train_feature = train_feature.to(device)
    train_label = train_label.to(device)
    test_feature = test_feature.to(device)
    test_label = test_label.to(device)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr = learning_rate, weight_decay = weight_decay)
    for epoch in range(num_epoch):
        for x,y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            l = binary_cross_entropy(net(x),y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # print(net(train_feature).cpu().device)
        train_loss = binary_cross_entropy(net(train_feature),train_label)
        test_acc = accuracy(net(test_feature),test_label)
        print('epoch = %.0f, train_loss = %.6f, test_acc = %.4f' % (epoch,train_loss,test_acc))



def main(args):
    # rank_zero_info(vars(args))
    # results = globals()["main_" + args.settings](args)
    adj = get_adj()
    train_dataset, test_dataset = get_feature_dataset(args)
    model = get_model(args,adj)
    train(model,train_dataset,test_dataset,args.num_epoch,args.learning_rate,args.weight_decay,args.batch_size)
    # return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("shenzhen", "losloop"), default="losloop"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="GCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--normalize", type=bool, default=True)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--num_layer", type=int, default=3)

    parser.add_argument("--learning_rate", type = float, default=1e-3)
    parser.add_argument("--weight_decay", type = float, default= 1.5e-3)
    parser.add_argument("--num_epoch", type = int, default=100)

    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    # temp_args, _ = parser.parse_known_args()
    args = parser.parse_args()
    # args, _ = parser.parse_known_args()
    main(args)
    # parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    # parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    # parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    # args = parser.parse_args()
    # utils.logging.format_logger(pl._logger)
    # if args.log_path is not None:
    #     utils.logging.output_logger_to_file(pl._logger, args.log_path)

    # try:
    #     results = main(args)
    # except:  # noqa: E722
    #     traceback.print_exc()
    #     if args.send_email:
    #         tb = traceback.format_exc()
    #         subject = "[Email Bot][❌] " + "-".join([args.settings, args.model_name, args.data])
    #         utils.email.send_email(tb, subject)
    #     exit(-1)
    #
    # if args.send_email:
    #     subject = "[Email Bot][✅] " + "-".join([args.settings, args.model_name, args.data])
    #     utils.email.send_experiment_results_email(args, results, subject=subject)