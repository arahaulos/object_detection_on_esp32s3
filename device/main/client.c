#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_event.h"
#include "nvs_flash.h"

#include "esp_netif.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>

#include "client.h"

#define MAX_TRANSFER_SIZE 256

void init_server_info(server_info *addr, const char *ip, int port)
{
    strcpy(addr->ip, ip);
    addr->port = port;
}

int connect_server(server_info *info)
{        
    struct sockaddr_in dest_addr;
    inet_pton(AF_INET, info->ip, &dest_addr.sin_addr);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(info->port);

    int sock =  socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        printf("Socket creation failed\n");
        return -1;
    }

    int err = connect(sock, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
    if (err != 0) {
        printf("Cannot connect\n");

        shutdown(sock, 0);
        closesocket(sock);

        return -1;
    }
    return sock;
}

void close_connection(int sock)
{
    shutdown(sock, 0);
    closesocket(sock);
}


void send_data(int sock, uint8_t *data, uint32_t data_len)
{
    send(sock, (void*)&data_len, sizeof(uint32_t), 0);
    while (data_len > 0) {
        uint32_t tx_size = data_len;
        if (tx_size > MAX_TRANSFER_SIZE) {
            tx_size = MAX_TRANSFER_SIZE;
        }
        send(sock, (void*)data, tx_size, 0);
        data = data + tx_size;
        data_len -= tx_size;
    }
}

void receive_data(int sock, uint8_t *data, uint32_t data_len)
{
    while (data_len > 0) {
        int bytes_to_receive = data_len;
        if (bytes_to_receive > MAX_TRANSFER_SIZE) {
            bytes_to_receive = MAX_TRANSFER_SIZE;
        }
        int received = recv(sock, data, bytes_to_receive, 0);
        data += received;
        data_len -= received;
    }
}


void send_string(int sock, const char *str)
{
    send(sock, (void*)str, strlen(str)+1, 0);
}

void receive_string(int sock, char *buffer, int buffer_size)
{
    int bytes_received = 0;
    while (bytes_received < buffer_size) {
        int received = recv(sock, &buffer[bytes_received], 1, 0);

        printf("%c\n", buffer[bytes_received]);

        if (buffer[bytes_received] == 0) {
            break;
        }
        bytes_received++;
    }
    buffer[buffer_size-1] = 0;
}


void send_image(server_info *addr, uint8_t *image_data, uint32_t image_data_len)
{
    int sock = connect_server(addr);
    if (sock == -1) {
        return;
    }

    send_string(sock, "upload_image");
    send_data(sock, image_data, image_data_len);

    close_connection(sock);
}

void send_bboxes(server_info *addr, bbox *bboxes, uint32_t num_of_bboxes)
{
    int sock = connect_server(addr);
    if (sock == -1) {
        return;
    }

    send_string(sock, "upload_bboxes");
    send_data(sock, (uint8_t*)bboxes, num_of_bboxes*sizeof(bbox));

    close_connection(sock);
}

uint32_t request_object_detection(server_info *addr, bbox *bboxes_buffer, uint32_t max_bboxes)
{
    int sock = connect_server(addr);
    if (sock == -1) {
        return 0;
    }

    send_string(sock, "request_object_detection");

    uint32_t num_of_bboxes = 0;
    receive_data(sock, (uint8_t*)&num_of_bboxes, sizeof(uint32_t));

    if (num_of_bboxes > max_bboxes) {
        num_of_bboxes = max_bboxes;
    }
    receive_data(sock, (uint8_t*)bboxes_buffer, num_of_bboxes*sizeof(bbox));

    close_connection(sock);

    return num_of_bboxes;
}


bool test_connection(server_info *addr)
{
    int sock = connect_server(addr);
    if (sock == -1) {
        return false;
    }

    send_string(sock, "ping");

    char buffer[32];

    receive_string(sock, buffer, 32);

    close_connection(sock);

    printf("%s\n", buffer);

    if (strcmp(buffer, "ping") == 0) {
        return true;
    } else {
        return false;
    }
}