# test BLE Scanning software
# jcs 6/8/2014

import blescan
import sys
import pymysql
import bluetooth._bluetooth as bluez
import time

dev_id = 0
try:
	sock = bluez.hci_open_dev(dev_id)
	print "ble thread started"

except:
	print "error accessing bluetooth device..."
    	sys.exit(1)

blescan.hci_le_set_scan_parameters(sock)
blescan.hci_enable_le_scan(sock)

while True:
        time.sleep(3)
        conn = pymysql.connect(host='192.168.1.65', user = 'root', password='sig101!@', db='project', charset='utf8')
        cursor = conn.cursor()
        sql0 = "UPDATE soldier SET connect = %s where connect = %s"
        cursor.execute(sql0,("X","O"))
	sql1 = "DELETE FROM Add_beacon"
        cursor.execute(sql1) 
        conn.commit()
        returnedList = blescan.parse_events(sock, 10)
	print "----------"
	for beacon in returnedList:
		sptdata = beacon.split(',')
                device = sptdata[0]
                uuid = sptdata[1]
                sig = sptdata[5]
                print "Dev : " + device + "\, UUID : " + uuid + "\, Signal : " + sig
                if int(sig) > -80:
                    print "Device : " + device
                    print "UUID : " + uuid
                    sql2 = "INSERT INTO add_beacon (no, mac, uuid) VALUES ( 1, %s, %s )"
                    cursor.execute(sql2,(device, uuid))
                    sql3 = "SELECT * FROM soldier where uuid = %s"
                    cursor.execute(sql3, (uuid))
                    rows = cursor.fetchall()
                    if not rows:
                        print "Not matched."
                    else:
                        sql4 = "UPDATE soldier SET connect = %s where uuid = %s"
                        cursor.execute(sql4,("O",uuid))
                        conn.commit()
                        print("Updated.")
                    sql5 = "SELECT * FROM soldier WHERE uuid=%s"
                    rows2 = cursor.fetchall()
                    print(rows2)
        conn.commit()
        conn.close()
