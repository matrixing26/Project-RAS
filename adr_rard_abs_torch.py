import numpy as np
#import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import deepxde as dde
import time
from scipy import interpolate
from torch.autograd.functional import hessian
#https://discuss.pytorch.org/t/linear-interpolation-in-pytorch/66861/9
#torch的线性插值
from datetime import datetime

def dirichlet(inputs, outputs):##########要根据deeponet和pinn写两种
    
    
    
    #print(type(inputs),"0")
    if isinstance(inputs,tuple): ####deeponet形式
        print("before",outputs)
        x_trunk = inputs[0]
    
        #x_trunk = x_trunk.reshape(-1,2)
    
        #print("x_trunk",x_trunk.shape)
        x, t = x_trunk[:, 0].reshape(-1,1), x_trunk[:, 1].reshape(-1,1)
       # print("outputs,x,t",outputs.shape,x.shape,t.shape)
        print("after:",2 * t * outputs + torch.sin(2 * np.pi * x))
        return 2 * t * outputs + torch.sin(2 * np.pi * x)
    else:####pinn形式，输入是tensor
      #  print(inputs,inputs.shape,outputs,outputs.shape)
        outputs = outputs.reshape(-1,1)
        x, t = inputs[:, 0].reshape(-1,1), inputs[:, 1].reshape(-1,1)
      #  print(x,t)
        outputs = 2 * t * outputs + torch.sin(2 * np.pi * x)
     #   print(outputs.reshape(1,-1))
        return outputs.reshape(1,-1)
    
    
    
    

def periodic(x):
    #print( "shape",torch.sin(x[:, 0] * 2 * np.pi).shape)
    return torch.cat(   ( torch.cos(x[:, 0] * 2 * np.pi).reshape(-1,1), torch.sin(x[:, 0] * 2 * np.pi).reshape(-1,1),
                      torch.cos(2 * x[:, 0] * 2 * np.pi).reshape(-1,1), torch.sin(2 * x[:, 0] * 2 * np.pi).reshape(-1,1), x[:, 1].reshape(-1,1)  ), 1)

    




def gelu(x):
    # return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3)))
    return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2))))

def get_data_test(filename):

    data = np.load(filename)
    u0 = data["X_test0"].astype(np.float32)
    xt = data["X_test1"].astype(np.float32)
    u = data["y_test"].astype(np.float32)  # N x nt x nx


    return (u0, xt), u

# def get_data(filename):

#     data = np.load(filename)
#     u0 = data["X_train0"].astype(np.float32)
#     xt = data["X_train1"].astype(np.float32)
#     u = data["y_train"].astype(np.float32)  # N x nt x nx


# #(1000, 100)
# #(10000, 2)
# #(1000, 10000)
# #dist[idx1, :]

#     return (u0, xt), u
def get_data(filename):
    nx = 100
    nt = 100
    data = np.load(filename)
    x = data["x"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["u"].astype(np.float32)  # N x nt x nx

    u0 = u[:, 0, :]  # N x nx
    xt = np.vstack((np.ravel(x), np.ravel(t))).T
    u = u.reshape(-1, nt * nx)
    return (u0, xt), u








def main():
    name_train_file = "train_IC2.npz"
    name_test_file = "test_IC2.npz"
    print("tet")
    train_scheme = [30000,50000]
    b_s = None
    choose_scheme = [30,2000]
    mode = "advection_RAS-D-abs_GRF"
    ini_datasize = 100
    ls = 0.05
    load_pretrain = False
    PATH = "./models/pre_train_ini" + str(ini_datasize) +"steps" + str(train_scheme[0]) + ".pth"    
    if load_pretrain == True:
        print("load pre-train")
    if load_pretrain == False:
        print("train raw")
    print("batch size",b_s)
    print(mode)    
    print("train_scheme",train_scheme,"choose scheme",choose_scheme)
    print("ini_datasize",ini_datasize)
    print("ls",ls)
    loss_list = []#最后把这个转成array然后输出
    
    
#prepare for model    
    nt = 100
    nx = 100

    data = np.load(name_train_file)
    x = data["x"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["u"].astype(np.float32)  # N x nt x nx

    u0 = u[:, 0, :]  # N x nx
    xt = np.vstack((np.ravel(x), np.ravel(t))).T
    u = u.reshape(-1, nt * nx)
    

    indice = np.arange(ini_datasize)


    x_train, y_train = get_data(name_train_file)
    x_test, y_test = get_data(name_test_file)
    
    u0_ini = u0[indice,:]
    xt_ini = xt
    u_ini  = u[indice,:]
    
    x_train_temp = (u0_ini,xt_ini)
    y_train_temp = u_ini
    x_test_temp = x_test#这里还是全部数据
    y_test_temp = y_test
    
    data_oprt = dde.data.TripleCartesianProd(x_train_temp, y_train_temp, x_test_temp, y_test_temp)

    net = dde.maps.DeepONetCartesianProd(
        [nx, 100, 100], [5, 100, 100, 100], gelu, "Glorot normal"#这里trunck input size是5是因为周期条件
    )
    net.apply_feature_transform(periodic)
    # net.apply_output_transform(dirichlet)

#pre-train
    if load_pretrain == True:
        net.load_state_dict(torch.load(PATH))
        print("successfully loaded!")
        



    if load_pretrain == False:
        model = dde.Model(data_oprt, net)
        model.compile(
            "adam",
            lr=5e-3,
            metrics=["mean l2 relative error"],
        )
        losshistory, train_state = model.train(epochs=train_scheme[0], batch_size=b_s)
        PATH1 = "./models/pre_train_ini" + str(ini_datasize) +"steps" + str(train_scheme[0]) + ".pth"
        torch.save(model.state_dict(), PATH1)
        flat_list = []
        for sublist in losshistory.metrics_test:
            for item in sublist:
                flat_list.append(item)
        
      #  loss_list.append(np.min(np.array(flat_list)))
    




#prepare for PINN data

    def pde(x, y):
        #print("yaode",net(x))
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        #dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        

        sensors2 = np.linspace(0, 1, num=sensor_value.shape[0])
        int2 = interpolate.interp1d(sensors2, sensor_value, kind='linear')
        D_x = torch.from_numpy(int2(x[:, 0].detach().cpu().numpy())[:, None].astype(np.float32)).cuda()
        
        # print("dy_xx",dy_xx.mean())
        # print("y",y.mean())
        # print("v_x",v_x.mean())
        # print("dy_t",dy_t.mean())            
        return dy_t + dy_x# - D_x * dy_xx
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    X = geomtime.random_points(10000)
    xt_fine2 = np.array([[a, b] for a in np.linspace(0.07, 0.93, 101) for b in np.linspace(0, 1, 101)])


    data_pde = dde.data.TimePDE(geomtime, pde, [], num_domain=2500, num_test=10000, train_distribution="pseudo")
    model = dde.Model(data_pde, net)
    model.compile(
            "adam",
            lr=1e-3,

            metrics=["mean l2 relative error"],
        )
    residual = []
    index = []
 #注意到data_pde是不变的，只需要compile一次

#aplying pinn


    for j in range(1,30):
        k = 2
        c = 0
        residual = []#每次清空residue
        #first select
        mean_all = []
        std_all = []

        temp_residue = 0#当做之前residue的和，用类似adam的算法
        for i in range(choose_scheme[1]):

            sensor_value = x_train[0][j*choose_scheme[1]+i,:]
            #print("sensor_value",sensor_value.shape)
            net.inputs = [sensor_value, None]
            residual.append( np.abs(model.predict(xt_fine2, operator=pde)).mean() )
            dde.grad.clear()

        #print("residues_oring",residual)
        mean_all.append(np.mean(np.array(residual)))
        std_all.append(np.std(np.array(residual)))
        print("mean and std of all",mean_all, std_all)
        print("max and min of pinn",np.max(np.array(residual)), np.min(np.array(residual) ))
        residual = np.power(residual, k) / np.power(residual, k).mean()
        # print(residual.shape)
        err_eq_normalized = (residual / sum(residual))
        print("max residues_prob",np.max(err_eq_normalized))
        print("min residues_prob",np.min(err_eq_normalized))
        new_indice = np.random.choice(
            a=np.array(range(j*choose_scheme[1],(j+1)*choose_scheme[1])), size=choose_scheme[0], replace=False, p=err_eq_normalized
        )
        n_r = []
        for n_ind in list(new_indice):

            sensor_value = x_train[0][n_ind,:]
            #print("sensor_value",sensor_value.shape)
            net.inputs = [sensor_value, None]
            n_r.append( np.abs(model.predict(X, operator=pde)).mean() )
            dde.grad.clear()
        mean_selected = []
        std_selected = []
        mean_selected.append(np.mean(np.array(n_r)))
        std_selected.append(np.std(np.array(n_r)))
        print("mean and std selected",mean_selected, std_selected)
        #print("residues_picked",n_r)
        
        print("new_indice",new_indice.shape)
        
        indice = np.hstack((indice,new_indice))
        print(indice,end=', ')
        print("indice", indice.shape)
        
     
        u0_ini = u0[indice,:]
        xt_ini = xt
        u_ini  = u[indice,:]
        
        # print("u0_ini",u0_ini,u0_ini.shape)
        # print("u_ini[0]",u_ini[0].reshape(10,-1),u_ini.shape)
        
        x_train_temp = (u0_ini,xt_ini)
        y_train_temp = u_ini
        x_test_temp = x_test
        y_test_temp = y_test
        
        data_oprt = dde.data.TripleCartesianProd(x_train_temp, y_train_temp, x_test_temp, y_test_temp)
            
        
        model = dde.Model(data_oprt, net)
        model.compile(
            "adam",
            lr=1e-3,
    
            metrics=["mean l2 relative error"],
        )
        # IC1
        # losshistory, train_state = model.train(epochs=100000, batch_size=None)
        # IC2
        losshistory, train_state = model.train(train_scheme[1], batch_size=b_s)
        
        
        PATH = "./models/adr_"+ str(mode) +"_" + str(j) + ".pth"
        print(PATH)
        torch.save(model.state_dict(), PATH)
        
        
        
        flat_list = []
        for sublist in losshistory.metrics_test:
            for item in sublist:
                flat_list.append(item)
        
        loss_list.append(np.min(np.array(flat_list)))
#save        
        
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")    
    print(loss_list) 
    if b_s == None:
        b_s = "whole"    
    np.save(f"./results/{mode}_k_{k}_c_{c}_ls_{ls}_ini_{ini_datasize}_b_s_{b_s}_train_{train_scheme[0]}_{train_scheme[1]}_choose{choose_scheme[0]}_{choose_scheme[1]}_{date}.npy",np.array(loss_list))
        
        
        
        
        #https://icode.best/i/19352241343692
        #取数据集可以参考这个

        # torch.cuda.empty_cache()
        #print("emptied")
        #print(residual)

        #then train
        #更新一下数据集，compile，train








if __name__ == "__main__":
    main()
