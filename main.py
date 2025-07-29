from Model import Model
import sys

if __name__ == '__main__':

    model = Model()
    if len(sys.argv)>1 and (sys.argv[1] or sys.argv[1] == 1):
        model.train()

    client_full_name = input("Provide client full name: ")
    client_age = input("Provide client age: ")
    purchase_history = input("Provide purchase history: ")
    discount_offer = input("Provide discount offer: ")

    print(model.inquire(client_full_name.strip(), client_age.strip(), purchase_history.strip(), discount_offer.strip()))
