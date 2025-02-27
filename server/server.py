from PIL import Image
import socket

from threading import Thread
import struct
import bbox
import io
import time

class server:

    def receive_bytes(self, sock, num_of_bytes):

        print("Receiving {} bytes".format(num_of_bytes))

        received = 0
        data = bytearray()
        while (received < num_of_bytes):
            bytes = sock.recv(num_of_bytes)
            data += bytes
            received += len(bytes)

        return data
    
    def receive_string(self, sock):
        data = bytearray()
        while self.running:
            bytes = sock.recv(1)
            if (bytes[0] == 0):
                break
            data += bytes
            
        return data.decode("utf-8")

    def receive_bboxes(self, sock):
        print("Receiving bboxes")

        HEADER_FORMAT = "<I"
        HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

        BBOX_FORMAT = "<Iffffff"
        BBOX_SIZE = struct.calcsize(BBOX_FORMAT)

        bbox_data_length, = struct.unpack(HEADER_FORMAT, self.receive_bytes(sock, HEADER_SIZE))

        num_of_bboxes = int(bbox_data_length/BBOX_SIZE)

        bbox_data = self.receive_bytes(sock, bbox_data_length)

        bboxes = []
        for i in range(num_of_bboxes):
            ot, co, ed, x, y, w, h = struct.unpack_from(BBOX_FORMAT, bbox_data, i*BBOX_SIZE)
            bboxes.append(bbox.bbox(ot, co, ed, x, y, w, h))

        return bboxes

    def receive_image(self, sock):
        print("Receiving image")

        HEADER_FORMAT = "<I"
        HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

        image_data_length, = struct.unpack(HEADER_FORMAT, self.receive_bytes(sock, HEADER_SIZE))

        print("Image size {}".format(image_data_length))

        image_data = self.receive_bytes(sock, image_data_length)

        return Image.open(io.BytesIO(image_data))
        

    def send_bboxes(self, sock, bboxes):
        print("Sending bboxes")

        HEADER_FORMAT = "<I"
        BBOX_FORMAT = "<Iffffff"

        sock.send(struct.pack(HEADER_FORMAT), (len(bboxes)))
        for bbox in bboxes:
            sock.send(struct.pack(BBOX_FORMAT), (bbox.object_type, bbox.confidence, bbox.estimated_distance, bbox.x, bbox.y, bbox.w, bbox.h))


    def close(self):
        self.running = False
        self.thread.join()

    def handle_connection(self, addr, sock):
        self.active_connections += 1

        print("Handling connection {}".format(addr))

        sock.setblocking(True)

        request = self.receive_string(sock)

        print(request)

        if (request == "upload_image"):
            image = self.receive_image(sock)
            if (image != None):
                self.last_received_image = image

        elif (request == "upload_bboxes"):
            bboxes = self.receive_bboxes(sock)
            if (bboxes != None):
                self.last_received_bboxes = bboxes
        elif (request == "request_object_detection"):
            if (self.last_received_image != None):
                bboxes = self.object_detector.detect(self.last_received_image)
                self.send_bboxes(sock, bboxes)

        else:
            print("unknown request")

        sock.close()

        self.active_connections -= 1

    def server_loop(self, ip, port):
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.bind((ip, port))
        tcp_socket.setblocking(False)
        tcp_socket.listen(0)

        while self.running:
            try:
                client_socket, addr = tcp_socket.accept()

                print("Accepting connection {}".format(addr))

                new_thread = Thread(target=self.handle_connection, args=(addr, client_socket))
                new_thread.start()
                self.active_threads.append(new_thread)
            except BlockingIOError:
                pass
            
    def __init__(self, ip, port, object_detector):
        self.running = True
        self.active_connections = 0
        self.last_received_image = None
        self.last_received_bboxes = None
        self.object_detector = object_detector

        self.active_threads = []
        self.thread = Thread(target = self.server_loop, args = (ip, port))
        self.thread.start()

    def get_last_received_image(self):
        return self.last_received_image
    
    def get_last_received_bboxes(self):
        return self.last_received_bboxes
