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

#define MAX_PACKET_SIZE 64

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
        if (tx_size > MAX_PACKET_SIZE) {
            tx_size = MAX_PACKET_SIZE;
        }
        send(sock, (void*)data, tx_size, 0);
        data = data + tx_size;
        data_len -= tx_size;
    }
}


void send_string(int sock, const char *str)
{
    send(sock, (void*)str, strlen(str)+1, 0);
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

void send_bboxes(server_info *addr, detected_bbox *bboxes, uint32_t num_of_bboxes)
{
    int sock = connect_server(addr);
    if (sock == -1) {
        return;
    }

    send_string(sock, "upload_bboxes");
    send_data(sock, (uint8_t*)bboxes, num_of_bboxes*sizeof(detected_bbox));

    close_connection(sock);
}

uint32_t request_object_detection(server_info *addr, detected_bbox *bboxes_buffer, uint32_t max_bboxes)
{
    return 0;
}