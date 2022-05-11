import socket
import camera
import virtualFitting
import recommedation
import sys

# python socket_android.py ip_address
host = sys.argv[1]
port = 9990

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(1)

while True:
    client_sock, addr = server_sock.accept()
    if client_sock:
        in_data = client_sock.recv(2048)
        print(in_data)
        input_message=in_data.decode("utf-8")
        input_message = input_message[2:]
        print(input_message)

        # take a picture user's upperbody/lowerbody(->user's clothes) to upload user's clothes into database
        if input_message[:3] == "cam":
            result = camera.capture()
            client_sock.send(("picture").encode("utf-8"))
            print('send : picture')
        # client send a message about virtual fitting - provide virtual fitting service
        elif input_message[:3] == "lin":
            # link
            link_str = input_message[3:]
            print(link_str)

            # get user's upper body and lower body coordinate to provide virtualfitting service
            result = camera.capture()

            link_str = link_str+"X="+str(result[0][0])+"&Y="+str(result[0][1])+"&w="+str(result[0][2])+"&h="+str(result[0][3])+"&lh="+str(result[1][3])
            virtualFitting.fitting(link_str)
            client_sock.send(("fitting").encode("utf-8"))
            print('send : fitting')
        # client send message that client wants to be provided recommendation service(by using user's clothes) - provide recommendation styling
        elif input_message[:3]=='rec':

            # set temperature_section appropriately
            temperature_section=5 # TODO: have to set temperature_section dynamically
            if temperature_section >= 3 and temperature_section <= 7:
                top_temperature_section = 3
                bottom_temperature_section = 3
            else:
                top_temperature_section = temperature_section
            if temperature_section == 1 or temperature_section == 2:
                bottom_temperature_section = 1

            # make a combination to recommend styling with user's clothes
            if temperature_section < 4:
                recommedation.top_bottom(top_temperature_section, bottom_temperature_section)
            else:
                recommedation.outer_top_bottom(temperature_section, top_temperature_section, bottom_temperature_section)
        client_sock.send(("recommend").encode("utf-8"))
        print('send : recommend')
client_sock.close()
server_sock.close()
