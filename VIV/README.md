# VIV(vortex induced vibration)涡激振动
This guide introduces to how to build a simple VIV model with PINNs in PaddleScience.



## Use case introduction
The current model is a typical inverse problem application, and the VIV system is equivalent to a one-dimensional spring-damper system,as figure below.

<div align="center">
<img src="image/VIV_1D_SpringDamper.png" width = "400" align=center />
</div>


The governing equation is as follows, and *λ1* and *λ2* represent the natural damping and stiffness of the structure, *ρ* is mass. In this inverse problem, getting the value of *λ1* and *λ2* makes our goal. 
<div align="center">
<img src="image/VIV_eq.png" width = "200" align=center />
</div>

In order to verify the correctness of the inverse problem based on PINNS, the real values of stiffness and damping of the system are obtained in advance（*λ1=0，λ2=1.093*）. Comparing the stiffness and damping predicted by the model with the real value, if the relative error is less than 5%, we believe that the model can well simulate the one-dimensional vibration phenomenon of VIV and predict the physical properties of the unknown structure, such as some compicated structure.

This model assumes a constant reduction velocity `Ur=8.5（Ur=u/(fn*d))`, corresponding to `Re=500`. The amplitude of cylinder vibration caused by the velocity fluid flowing through the cylinder and the corresponding lift force（*f*） are recorded. These known data serve as the monitoring data of the training neural network and  are combined with the governing equation to form the total loss of the net.

## How to run this model

### Confirm the governing equation 
The equation needs to be defined, as described earlier.

### Define the Network
Since only the lateral vibration of the structure is considered and the constant inlet velocity is given, the input of the network is only time(*t*), and the output is the vibration amplitude of the structure(*η*),


## How to run this model
