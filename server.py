from server_obj import Server


if __name__ == "__main__":
    serv_obj = Server("10.42.0.1", 9999)
    serv_obj.execute()
