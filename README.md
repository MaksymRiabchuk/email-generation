___Brief description___
-
This model is trained to create promotional emails based on given client information and purchase history.

There are two main method in Model class, the first one is ___train___ which responsible for training model.
Model analyzes data written in file which is passed by the parameter _dataset_path_(default dataset.json) and after the training is finished creates _gpt-model_ directory
where all information about model is stored. The second method ___inquire___ is used for testing the model, basically getting wanted emails
by giving the client info and purchase history. 

Main script can be run with parameter ___run_with_training___, if run the main script with parameter ___True or 1___,
model will start training and after the training is finished, you can provide client data.

___Input parameters:___
- 
 __Client name__ - should be two words like this ___John Smith___

 __Client age__ - should be a number between 18 and 100

 __Purchase history__ - should be a sequence of products with date of purchase and number of bought products, separated by commas example:

2024-12-04 - Keyboard price of one - 5.92 x 1 pcs,

2024-08-22 - Gaming keyboard price of one - 88.80 x 1 pcs,

2024-08-03 - Processor i5 Intel price of one - 146.83 x 1 pcs,

2024-08-03 - Motherboard ASUS price of one - 99.19 x 2 pcs,

2024-07-21 - Gaming mouse price of one - 36.70 x 2 pcs,

2024-05-16 - Keyboard price of one - 7.36 x 3 pcs,

2024-05-16 - Mouse price of one - 8.62 x 1 pcs

_Important notes_
-
Date of purchase and amount of products are very significant due to the reason that model decides the best matching products based on these values, while the price of products is not that important.

Model gives the output email with given input. Sometimes model can give a little bit of odd answers, due to the lack of training on real data, quite small amount of data and short time of training.  

___Example Input:___
-
Provide client full name: John Smith
Provide client age: 32
Provide purchase history: 2024-12-28 - SSD 512GB price of one - 80.72 x 1 pcs, 2024-10-01 - Processor i5 Intel price of one - 99.68 x 1 pcs, 2024-10-01 - Gaming keyboard price of one - 82.62 x 3 pcs, 2024-05-28 - Processor i5 Intel price of one - 124.61 x 3 pcs, 2024-01-28 - Webcam HD price of one - 54.96 x 1 pcs, 2023-11-11 - RAM 32GB price of one - 89.98 x 1 pcs, 2023-09-07 - Gaming graphic card price of one - 1344.81 x 1 pcs
Provide discount offer: 15% discount on similar products

___Output:___
-
Instruction: Generate a promotional email for the given client based on their purchase history and discount offer.
Input: Client name: John Smith
Client age: 32 y.o.
Purchase history: 2024-12-28 - SSD 512GB price of one - 80.72 x 1 pcs, 2024-10-01 - Processor i5 Intel price of one - 99.68 x 1 pcs, 2024-10-01 - Gaming keyboard price of one - 82.62 x 3 pcs, 2024-05-28 - Processor i5 Intel price of one - 124.61 x 3 pcs, 2024-01-28 - Webcam HD price of one - 54.96 x 1 pcs, 2023-11-11 - RAM 32GB price of one - 89.98 x 1 pcs, 2023-09-07 - Gaming graphic card price of one - 1344.81 x 1 pcs
Discount offer: 15% discount on similar products

__Good afternoon, John Smith.
We noticed that you recently purchased SSD 512GB. This week, we are pleased to offer you a special 15% discount on similar products. Please remember to use your frequent-buyer coupon — it remains valid until Sunday.
Best regards__

___Required libraries___
-

- PyTorch
- Datasets
- Transformers
- Accelerate

___Optional for GPU using:___
-
- Torchvision
- Torchaudio
