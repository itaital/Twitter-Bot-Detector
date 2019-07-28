
# define functionality inside class
from keras.callbacks import Callback


class GraphIndex:
    """
        Enum to readable indexes of the graph array
    """
    ACC_EPOCH  = 0
    LOSS_EPOCH = 1
    ACC_BATCH  = 2
    LOSS_BATCH = 3


class AccuracyHistory(Callback):
    """
    CallBack to print log and draw the graphs
    """
    def __init__(self, graph, frame, log_print, graph_draw, epoch):
        super(AccuracyHistory, self).__init__()
        self.graph_arr = graph
        frame.setVisible(True)
        self.index_on_epoch = self.index_on_batch = 0
        self.index_log_on_batch = [0]
        self.index_log_on_epoch = [0]
        self.log_print = log_print
        self.draw = graph_draw
        self.epoch = epoch
        self.logs = [[0], [0], [0], [0]]

    def on_epoch_begin(self, epoch, logs=None):
        """
        clear data before epoch begin
        """
        self.index_log_on_batch = [0]
        self.index_on_batch = 0
        self.logs[GraphIndex.ACC_BATCH] = [self.logs[GraphIndex.ACC_BATCH][-1]]
        self.logs[GraphIndex.LOSS_BATCH] = [self.logs[GraphIndex.LOSS_BATCH][-1]]
        self.graph_arr[GraphIndex.ACC_BATCH].clear()
        self.graph_arr[GraphIndex.LOSS_BATCH].clear()
        self.log_print[str].emit("Epoch number {} of {}".format(epoch+1, self.epoch))

    def on_epoch_end(self, batch, logs={}):
        """
        print log and graph data on epoch end
        """
        self.index_on_epoch +=1
        self.index_log_on_epoch.append(self.index_on_epoch)
        self.logs[GraphIndex.ACC_EPOCH].append(logs.get('acc'))
        self.logs[GraphIndex.LOSS_EPOCH].append(logs.get('loss'))
        self.draw.emit(self.graph_arr[GraphIndex.ACC_EPOCH], self.logs[GraphIndex.ACC_EPOCH], self.index_log_on_epoch)
        self.draw.emit(self.graph_arr[GraphIndex.LOSS_EPOCH], self.logs[GraphIndex.LOSS_EPOCH], self.index_log_on_epoch)
        self.log_print[str].emit("Epoch training result: acc: {0:.4f} loss:{1:.4f}".format(logs.get('acc'), logs.get('loss')))
        #self.log_print[str].emit("Epoch validation result: acc: {0:.4f} loss:{1:.4f}\n\n".format(logs.get('val_acc'), logs.get('val_loss')))

    def on_batch_end(self, batch, logs=None):
        """
        print log and graph data on batch end
        """
        self.index_on_batch += 1
        self.index_log_on_batch.append(self.index_on_batch)
        # calculate new accumulative average
        avg_acc = ((self.index_on_batch-1) * self.logs[GraphIndex.ACC_BATCH][-1] + logs.get('acc')) / self.index_on_batch
        avg_loss = ((self.index_on_batch-1) * self.logs[GraphIndex.LOSS_BATCH][-1] + logs.get('loss')) / self.index_on_batch
        self.logs[GraphIndex.ACC_BATCH].append(avg_acc)
        self.logs[GraphIndex.LOSS_BATCH].append(avg_loss)
        self.draw.emit(self.graph_arr[GraphIndex.ACC_BATCH], self.logs[GraphIndex.ACC_BATCH], self.index_log_on_batch)
        self.draw.emit(self.graph_arr[GraphIndex.LOSS_BATCH], self.logs[GraphIndex.LOSS_BATCH], self.index_log_on_batch)
        self.log_print[str].emit("acc: {0:.4f} loss:{1:.4f}".format(avg_acc, avg_loss))
