# Data Challenge: Catalog Categorization

This is a major modification of the previous "Catalog Categorization" challenge.
We noted that very few candidates chose to try this challenge. Most went for the "conversion prediction".

We decided to recreate the catalog categorization challenge with fewer categories and a cleaner dataset.
We also downloaded the images and provided tar.gz files with the product images in multiple resolutions,
including 224x224, which matches some known pretrained (ImageNet) CNNs.
All resolutions were rescaled from the original 75x75 images.

I have no idea if there is a reasonable solution for this problem without using the images.
I have no idea if there is a reasonable solution for this problem even using the images.


## Training dataset

**product_id**: ...
**product_name**: ...
**brand**: ...
**product_description**: ...
**gender**: ...
**color**: ...
**price**: ...
**division**: ...
**box_weight**: ...
**product_large_image**: URL to a larger image of the product (we cannot garantee all images will be available at the time of access)
**position_in_stock**: ...
**category**: Product category (target)


## Test dataset

There is a separate test dataset, in which all values in the column "category" were replaced by "PREDICT_CATEGORY".
There is also an additional "category_hash", which is a SHA1 of CONCAT(SALT,product_id,category,PEPPER).
It should be made clear that the candidates will not be evaluated by their accuracy on the test dataset.

There is a check_predictions.ipynb, which can kind of "dehash" and check the candidate predictions (it generates a confusion matrix).
We can show this to the candidate at the interview and see his reasoning there.

## Some dark secrets

**Do not share this info with candidates.**
product_name is the string of the product name. When the category appeared in the title, it was replaced by "Produto". When the brand appeared, it was replaced by the anonymized brand.
brand: brands were anonymized. The most frequent brands were replaced by Star Wars names. When we ran out of names, they were replaced by "<BRAND_000000>", "<BRAND_000001>", ...
product_description: same thing as product_name.
price: original_price * random_discount_between(0.7, 0.9)
box_weight = A * correct_wight + B, for whatever A and B
position_in_stock: BS variable. It is a random between 'Setor A' and 'SETOR Q'. The candidate should ignore this or the trained model should not depend strongly on this variable, otherwise there is likely overfitting.
