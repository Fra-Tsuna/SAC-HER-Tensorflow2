from mpi4py import MPI
import numpy as np
import tensorflow as tf

def sync_networks(network):
    variables = network.variables

    # broadcast variables
    if len(variables) > 0:
        comm = MPI.COMM_WORLD
        flat_variables = np.concatenate([np.array(var).flatten() for var in variables])
        #flat_variables = comm.Bcast(flat_variables, root=0)
        comm.Bcast(flat_variables, root=0)

        # variables reconstruction and assignment
        index = 0
        for var, tgt_var in zip(variables, network.variables):
            shape = np.array(var).shape
            new_var = None
            if len(shape) == 1:
                new_var = tf.convert_to_tensor(flat_variables[index:(index + shape[0])])
                index += shape[0]
            else:
                new_var = tf.convert_to_tensor(
                    np.reshape(flat_variables[index:(index + shape[0]*shape[1])], shape)
                )
                index += shape[0]*shape[1]
            tgt_var.assign(new_var)

def sync_grads(network, gradients):
    flat_grads = np.concatenate([np.array(grad).flatten() for grad in gradients])
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)

    # Gradients reconstruction
    index = 0
    new_grads = []
    for grad in gradients:
        shape = np.array(grad).shape
        if len(shape) == 1:
            new_grads.append(tf.convert_to_tensor(global_grads[index:(index + shape[0])]))
            index += shape[0]
        else:
            new_grads.append(tf.convert_to_tensor(
                np.reshape(global_grads[index:(index + shape[0]*shape[1])], shape)
            ))
            index += shape[0]*shape[1]
    return new_grads
