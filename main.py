from typing import Optional, Callable, Tuple
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor_ops as ops


class EdgeEncoder(tf.keras.layers.Layer):
    """
    simple edge encoder make the dim of edge and node consistent
    """

    def __init__(self, emb_dim):
        super(EdgeEncoder, self).__init__()

        self.dense1 = tf.keras.layers.Dense(2 * emb_dim)

        self.dense2 = tf.keras.layers.Dense(emb_dim)

    def call(self, h):
        h = self.dense1(h)
        h = self.dense2(h)

        return h


class GINConv(tfgnn.keras.layers.AnyToAnyConvolutionBase):
    def __init__(
            self,
            message_fn: tf.keras.layers.Layer,
            node_feature_dim: int,
            reduce_type: str = "sum",
            *,
            combine_type: str = "concat",
            receiver_tag: const.IncidentNodeTag = const.TARGET,
            receiver_feature: const.FieldName = const.HIDDEN_STATE,
            sender_node_feature: Optional[
                const.FieldName] = const.HIDDEN_STATE,
            sender_edge_feature: Optional[const.FieldName] = None,
            debug=False,
            **kwargs):
        super().__init__(
            receiver_tag=receiver_tag,
            receiver_feature=receiver_feature,
            sender_node_feature=sender_node_feature,
            sender_edge_feature=sender_edge_feature,
            **kwargs)

        self._message_fn = message_fn
        self._reduce_type = reduce_type
        self._combine_type = combine_type
        self.node_feature_dim = node_feature_dim
        self.edge_encoder = EdgeEncoder(emb_dim=self.node_feature_dim)

    def get_config(self):
        return dict(
            message_fn=self._message_fn,
            reduce_type=self._reduce_type,
            combine_type=self._combine_type,
            **super().get_config())

    def convolve(self, *,
                 sender_node_input: Optional[tf.Tensor],
                 sender_edge_input: Optional[tf.Tensor],
                 receiver_input: Optional[tf.Tensor],
                 broadcast_from_sender_node: Callable[[tf.Tensor], tf.Tensor],
                 broadcast_from_receiver: Callable[[tf.Tensor], tf.Tensor],
                 pool_to_receiver: Callable[..., tf.Tensor],
                 training: bool) -> tf.Tensor:
        # Collect inputs, suitably broadcast.
        inputs = []
        if sender_edge_input is not None:
            inputs.append(self.edge_encoder(sender_edge_input))
        if sender_node_input is not None:
            inputs.append(broadcast_from_sender_node(sender_node_input))
        if receiver_input is not None:
            inputs.append(broadcast_from_receiver(receiver_input))
        # Combine inputs.
        combined_input = ops.combine_values(inputs, self._combine_type)

        # Compute the result.
        messages = self._message_fn(combined_input)
        pooled_messages = pool_to_receiver(messages, reduce_type=self._reduce_type)

        return pooled_messages


class GINNodeUpdate(tf.keras.layers.Layer):
    def __init__(self, node_dim, debug=False):
        super(GINNodeUpdate, self).__init__()

        self.eps = tf.Variable(tf.zeros(1))
        self.mlp = MLP(node_dim=node_dim)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs: Tuple[
        const.FieldOrFields, const.FieldsNest, const.FieldsNest]):
        node, edge, _ = inputs

        h = self.mlp((1 + self.eps) * node + edge['edge'])

        h = self.bn(h)

        return h


class MLP(tf.keras.layers.Layer):
    def __init__(self, node_dim):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(2 * node_dim)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.dense2 = tf.keras.layers.Dense(node_dim)

    def call(self, x):
        h = self.dense1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dense2(h)
        return h


class GIN(tf.keras.Model):

    def __init__(self, node_dim, debug=False):
        super(GIN, self).__init__()
        self.node_dim = node_dim
        self.relu = tf.keras.layers.Activation('relu')
        self.debug = debug

        self.model = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                'node': tfgnn.keras.layers.NodeSetUpdate(
                    edge_set_inputs={'edge': GINConv(
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        node_feature_dim=self.node_dim,
                        receiver_feature=None,
                        message_fn=self.relu,
                        reduce_type="sum",
                        combine_type='sum',
                        receiver_tag=tfgnn.TARGET,
                        debug=self.debug,
                    )},

                    next_state=GINNodeUpdate(node_dim=self.node_dim, debug=self.debug),
                )
            }
        )

    def call(self, batched_data):
        out = self.model(batched_data)
        return out
