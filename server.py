from server_obj import Server


if __name__ == "__main__":
    server_obj = Server("10.42.0.1", 9999)
    server_obj.exec()
