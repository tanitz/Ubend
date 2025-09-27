
from pymodbus.client.sync import ModbusSerialClient

class PLCModbus:
    def __init__(self):
        # Initialize Modbus client
        self.client = ModbusSerialClient(
            method='rtu',
            port='/dev/ttyACM0',
            baudrate=38400,
            bytesize=8,
            stopbits=1,
            parity='N',
            timeout=1
        )
        self.client.connect()

        self.unit_id = 1
        self.loop()



    def loop(self):
 
            rr = self.client.read_holding_registers(20, 1, unit=self.unit_id)
            if not rr.isError():
                regs = rr.registers
                print(f"PLC Status: {regs[0]}")
            else:
                print('Failed to read PLC registers')

            self.rate.sleep()

if __name__ == '__main__':
    try:
        PLCModbus()
    except :
        pass