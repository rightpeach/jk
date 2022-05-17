# VIV(vortex induced vibration)涡激振动
This guide introduces to how to build a simple VIV model with PINNs.The current model is a typical inverse problem application, and the VIV system is equivalent to a one-dimensional spring-damper system,as figure below.

<div align="center">
<img src="../../docs/source/img/ldc2d_u_100x100.png" width = "500" align=center />
</div>


corresponding to the following equation:



Based on the vibration amplitude and lift data of the structure under the known flow velocity, the stiffness and damping of the structure are predicted in reverse by combining with the governing equation.
