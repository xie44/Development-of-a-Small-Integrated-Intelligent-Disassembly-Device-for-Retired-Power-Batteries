# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 09:13:46 2022

@author: wuhaijun

explain：The API was developed for XSEL-TT devices But not limited to the device  
"""
import serial
import time
TTURAT = 0 #串口设备全局
SerialData = "" #读取的数据
SendData = "" #发送数据暂存空间
AllLocation = ['X', 'Y', 'Z'] #全轴坐标储存

#连接设备 portx 串口号 bps 波特率 timeout 超时时间 None为无限
def Link(portx, bps, timeout):
    global TTURAT
    try:
        # 打开串口，并得到串口对象
        TTURAT = serial.Serial(portx, bps, timeout=timeout)
        Test_Call()
    except Exception as e:
        print("A serial port abnormal：", e)
        return False
        
#返回值读取
def serload(ser):
    while True:
        if ser.in_waiting:
            str1 = ser.readline().decode('GBK')  # 读一行，以/n结束。
            #print("return：",str1) 
            return str1
            break
        
#只返回是否发送完成信号
def Write_simple(ser):
    TTURAT.write(ser.encode("gbk"))
    SerialData = serload(TTURAT)
    if SerialData[0] == "#":
        print("Executed")
        return True
    else:
        print("Exception")
        return False
    
#设备连通性测试
def Test_Call():
    TTURAT.write("!992000123456789@@\r\n".encode("gbk"))  
    SerialData = serload(TTURAT)
    if SerialData[0] == "#":
        print("Connected")
        return True
    else:
        print("Connection exception!!!")
        return False

#设备下线 mode = 0：仅释放串口设备 1：释放串口设备并且所以轴下使能  2：释放串口设备并且所以轴回零下使能
def Downline(mode):
    if mode == 0:
        TTURAT.close() # 关闭串口 
        return True
    elif mode == 1:
        AxleEnabled(7,0)
        time.sleep(20)
        TTURAT.close() # 关闭串口 
        return True

#解除报警 
def ALARMReset():
    SendData = "!99252@@\r\n"
    Status = Write_simple(SendData)
    return Status

#软件复位 
def Reboot():
    SendData = "!9925B@@\r\n"
    TTURAT.write(SendData.encode("gbk"))
       
#轴使能 Number轴编号 EN 使能状态
def AxleEnabled(Number,EN):
    SendData = "!99232" + str(Number).zfill(2) + str(EN) + "@@\r\n"
    Status = Write_simple(SendData)
    return Status

#轴归零 Number轴编号
def AxleToZero(Number):
    SendData = "!99233" + str(Number).zfill(2) + "040000@@\r\n"
    Status = Write_simple(SendData)
    return Status

#轴动作（绝对位置） Number轴编号 Acceleration加速度 Deceleration减速度 Speed移动速度 Coord坐标
def AxleMoveAbsolute(Number ,Acceleration ,Deceleration ,Speed ,Coord):
    SendData = "!99234" + str(Number).zfill(2) + str(hex(Acceleration)[2:]).zfill(4) + str(hex(Deceleration)[2:]).zfill(4) + str(hex(Speed)[2:]).zfill(4) + str(hex(Coord)[2:]).zfill(8)+ "@@\r\n"
    Status = Write_simple(SendData)
    return Status   

#轴动作（相对位置） Number轴编号 Acceleration加速度 Deceleration减速度 Speed移动速度 Coord坐标
def AxleMoveRelative(Number ,Acceleration ,Deceleration ,Speed ,Coord):
    SendData = "!99235" + str(Number).zfill(2) + str(hex(Acceleration)[2:]).zfill(4) + str(hex(Deceleration)[2:]).zfill(4) + str(hex(Speed)[2:]).zfill(4) + str(hex(Coord)[2:]).zfill(8)+ "@@\r\n"
    Status = Write_simple(SendData)
    return Status 

#点动操作 Number轴编号 Acceleration加速度 Deceleration减速度 Speed移动速度 Distanc移动距离 Direction方向
def AxleMovePTP(Number ,Acceleration ,Deceleration ,Speed ,Distanc ,Direction):
    SendData = "!99236" + str(Number).zfill(2) + str(hex(Acceleration)[2:]).zfill(4) + str(hex(Deceleration)[2:]).zfill(4) + str(hex(Speed)[2:]).zfill(4) + str(hex(Distanc)[2:]).zfill(8)+ str(Direction) +"@@\r\n"
    Status = Write_simple(SendData)
    return Status 

#全轴位置查询 
def AllAxleQuery():
    SendData = "!9921207@@\r\n"
    TTURAT.write(SendData.encode("gbk"))
    SerialData = serload(TTURAT)
    for i in range(0,3):
        AllLocation[i] = int(SerialData[(16+i*16):(24+i*16)], 16)
    return AllLocation