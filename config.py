import numpy as np
import socket,time,os

class cfg:
    # dataset parameters (default)
    db_path = "/home/maksym/Desktop/SLT/data"
    out_path = "/home/maksym/Desktop/SLT/plots"
    f_shape = [16,16,3]
    train_samples = 10000
    test_samples = 100000
    train_steps = 20000
    lr = 0.001
    noise = None

    # hyperband parameters
    batch_size = 100

    # square exhaustive plot parameters
    n_repeats = 20
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)


class cfg01:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg11"
    out_path = "/mas/u/mkkr/mk/slt/cfg11"
    f_shape = [64,32,1]
    train_samples = 10
    test_samples = 10000
    train_steps = 10000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 2
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)


class cfg11:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg11"
    out_path = "/mas/u/mkkr/mk/slt/cfg11"
    f_shape = [64,32,1]
    train_samples = 10
    test_samples = 10000
    train_steps = 20000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 50
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)


class cfg12:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg12"
    out_path = "/mas/u/mkkr/mk/slt/cfg12"
    f_shape = [64,32,1]
    train_samples = 100
    test_samples = 10000
    train_steps = 20000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 50
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)




class cfg13:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg13"
    out_path = "/mas/u/mkkr/mk/slt/cfg13"
    f_shape = [64,32,1]
    train_samples = 1000
    test_samples = 10000
    train_steps = 20000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 50
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)


class cfg14:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg21"
    out_path = "/mas/u/mkkr/mk/slt/cfg21"
    f_shape = [64,32,1]
    train_samples = 10000
    test_samples = 10000
    train_steps = 20000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 50
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)



class cfg21:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg21"
    out_path = "/mas/u/mkkr/mk/slt/cfg21"
    f_shape = [64,64,1]
    train_samples = 2000000
    test_samples = 100000
    #train_steps = 100000
    lr = 0.001
    noise = None
    # hyperband parameters
    train_steps = [2000000, 200000, 20000, 2000,200, 20]
    noises = [0,1,2,4,8,16,32]
    batch_sizes = [1,10,100,1000,10000,100000]
    lrs = [0.00001,0.0001,0.001,0.01,0.1,1]

    # square exhaustive plot parameters
    n_repeats = 50
    l2_val = 64
    l3_val = 1


# change the number of training samples [10,100,1000,10000,100000]
# change the activation from ReLU to sigmoid (later)

# change the batch size [1,10,100,1000,10000,100000] (for an infinitely-sized dataset) (and lr, and num_steps) 64,64,1
# add noise [0,1,2,4,8,16,32] to Y



class cfg31:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg31"
    out_path = "/mas/u/mkkr/mk/slt/cfg31"
    f_shape = [64,32,1]
    train_samples = 10
    test_samples = 10000
    train_steps = 20000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 20
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)


class cfg32:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg32"
    out_path = "/mas/u/mkkr/mk/slt/cfg32"
    f_shape = [64,32,1]
    train_samples = 100
    test_samples = 10000
    train_steps = 20000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 20
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)


class cfg33:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg33"
    out_path = "/mas/u/mkkr/mk/slt/cfg33"
    f_shape = [64,32,1]
    train_samples = 1000
    test_samples = 10000
    train_steps = 20000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 20
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)


class cfg34:
    # dataset parameters (default)
    db_path = "/mas/u/mkkr/mk/slt/cfg34"
    out_path = "/mas/u/mkkr/mk/slt/cfg34"
    f_shape = [64,32,1]
    train_samples = 10000
    test_samples = 10000
    train_steps = 20000
    lr = 0.001
    noise = None
    # hyperband parameters
    batch_size = 100
    # square exhaustive plot parameters
    n_repeats = 50
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)


class cfg4_1:
    name = "cfg4_1"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    #db_path = os.path.join(db_path)
    out_path = os.path.join(out_path,name)

    genf_shape = [16,16,3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-3]
    n_runs = 10
    n_iterations= 10000
    batch_size = 100
    fun_shape=[16, 16, 16, 3]
    em_shape=[120, 60, 256]


class cfg4_6:
    name = "cfg4_6"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 10000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [120, 60, 256]


class cfg4_7:
    name = "cfg4_7"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 10000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [1, 1, 256]


class cfg4_8:
    name = "cfg4_8"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 10000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [30, 20, 256]


class cfg4_9:
    name = "cfg4_9"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 10000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [240, 120, 256]


class cfg4_10:
    name = "cfg4_10"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 10000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [240, 240, 256]



class cfg4_11:
    name = "cfg4_11"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    #db_path = os.path.join(db_path)
    out_path = os.path.join(out_path,name)

    genf_shape = [16,16,3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-4]
    n_runs = 10
    n_iterations= 20000
    batch_size = 100
    fun_shape=[16, 16, 16, 3]
    em_shape=[120, 60, 256]


class cfg4_12:
    name = "cfg4_12"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    #db_path = os.path.join(db_path)
    out_path = os.path.join(out_path,name)
    genf_shape = [16,16,3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-4]
    n_runs = 10
    n_iterations= 20000
    batch_size = 100
    fun_shape=[16, 16, 16, 3]
    em_shape=[1, 1, 256]


class cfg4_13:
    name = "cfg4_13"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    #db_path = os.path.join(db_path)
    out_path = os.path.join(out_path,name)
    genf_shape = [16,16,3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-4]
    n_runs = 10
    n_iterations= 20000
    batch_size = 100
    fun_shape=[16, 16, 16, 3]
    em_shape=[30, 20, 256]
    
    
class cfg4_14:
    name = "cfg4_14"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    #db_path = os.path.join(db_path)
    out_path = os.path.join(out_path,name)
    genf_shape = [16,16,3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-4]
    n_runs = 10
    n_iterations= 20000
    batch_size = 100
    fun_shape=[16, 16, 16, 3]
    em_shape=[240, 120, 256]
    
    
class cfg4_15:
    name = "cfg4_15"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_1"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    #db_path = os.path.join(db_path)
    out_path = os.path.join(out_path,name)

    genf_shape = [16,16,3]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-4]
    n_runs = 10
    n_iterations= 20000
    batch_size = 100
    fun_shape=[16, 16, 16, 3]
    em_shape=[240, 240, 256]


class cfg4_16:
    name = "cfg4_16"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_16"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [1, 256]


class cfg4_17:
    name = "cfg4_17"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_16"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [64, 256]


class cfg4_18:
    name = "cfg4_18"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_16"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [256, 256]


class cfg4_19:
    name = "cfg4_19"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_16"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 3]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 3]
    em_shape = [1024, 256]



class cfg4_20:
    name = "cfg4_20"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em_shape = [1, 256]


class cfg4_21:
    name = "cfg4_21"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em_shape = [256, 256]


class cfg4_22:
    name = "cfg4_22"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_22"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 16]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 16]
    em_shape = [1, 256]


class cfg4_23:
    name = "cfg4_23"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_22"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    genf_shape = [16, 16, 16]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 16]
    em_shape = [256, 256]


class cfg4_24:
    name = "cfg4_24"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [1, 256]


class cfg4_25:
    name = "cfg4_25"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [64, 256]


class cfg4_26:
    name = "cfg4_26"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [256, 256]



class cfg4_27:
    name = "cfg4_27"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [1, 256, 256]


class cfg4_28:
    name = "cfg4_28"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [64, 256, 256]


class cfg4_29:
    name = "cfg4_29"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 25000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [256, 256, 256]


class cfg4_30:
    name = "cfg4_30"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [1, 256]

    # scheduler
    scheduler = "dlp"
    # optimizer
    optimizer = "tf.train.GradientDescentOptimizer"


class cfg4_31:
    name = "cfg4_31"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [256, 256]

    # scheduler
    scheduler = "dlp"
    # optimizer
    optimizer = "tf.train.GradientDescentOptimizer"

class cfg4_32:
    name = "cfg4_32"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [1, 256]

    # scheduler
    scheduler = "none"
    # optimizer
    optimizer = "tf.train.AdamOptimizer"

class cfg4_33:
    name = "cfg4_33"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [256, 256]

    # scheduler
    scheduler = "none"
    # optimizer
    optimizer = "tf.train.AdamOptimizer"


class cfg4_34:
    name = "cfg4_34"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [1, 256]

    # scheduler
    scheduler = "none"
    # optimizer
    optimizer = "tf.train.RMSPropOptimizer"

class cfg4_35:
    name = "cfg4_35"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [256, 256]

    # scheduler
    scheduler = "none"
    # optimizer
    optimizer = "tf.train.RMSPropOptimizer"



class cfg4_36:
    name = "cfg4_36"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [1, 256]

    # scheduler
    scheduler = "none"
    # optimizer
    optimizer = "tf.train.AdadeltaOptimizer"


class cfg4_37:
    name = "cfg4_37"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [256, 256]

    # scheduler
    scheduler = "none"
    # optimizer
    optimizer = "tf.train.AdadeltaOptimizer"



class cfg4_38:
    name = "cfg4_38"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [1, 256]

    # scheduler
    scheduler = "none"
    # optimizer
    optimizer = "tf.train.GradientDescentOptimizer"


class cfg4_39:
    name = "cfg4_39"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg4_20"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
        out_path = "/mas/u/mkkr/mk/slt/"
    else:
        raise Exception("path not set up on machine")
    # db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-3]
    n_runs = 10
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em = "actcentron_embedding"
    em_shape = [256, 256]

    # scheduler
    scheduler = "none"
    # optimizer
    optimizer = "tf.train.GradientDescentOptimizer"


# embed layer 2 of the network
# I need to initialize 0-sized embedding for the original problem to prov that stuff works!
# change the loss from L2, to L2 on one-hot
# change the shape to MNIST
# change the data to MNIST