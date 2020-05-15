# Amazon Product Reviews Abstractive summaries

In total, the dataset consists of 60 annotated products from the 4 categories listed below (15 per each). Each product has 3 summaries written by different professional human annotators.
For annotation, we selected reviews that were written in English and were at least 45 words long. We also assured diversity of summaries by not allowing more than 10 submissions per annotator.

## Categories
* Electronics
* Clothing, Shoes and Jewelry
* Home and Kitchen
* Personal Care

If you find the dataset useful, please cite the paper.

## Summary Examples

> These transition tights are perfect for children sensitive to the tight sensation other tights have around the foot.  The material is soft and durable; they stand up well to both the rough nature of children, and the washing machine.  This product does tend to run slightly small, so purchasing one size up is recommended.

> This heart rate monitor is very user friendly for beginners.  Very easy to use straight out of the box. However, this product comes with an instruction booklet and internet instructions if you need help with initial set up. You can program your target heart rate and there is an audible beep when you are in or fall out of target range. Very easy to read display which can also has a backlight. The only drawback is the log will only store the last workout data.  Overall a very nice beginner heart rate monitor.

> Easy, convenient way to prevent humidity and moisture in small areas and containers such as safes, cabinets, jewelry boxes, and more. Features a color indicator to notify you when the pack has reached capacity. Easy to recharge by placing inside of an oven to dry the crystals out again. Doesn't work well in large areas, even with multiple packages.

## Data Annotation Process

We used the Amazon Mechanical Turk platform by hiring workers who satisfy the qualifications below. Each worker was asked to read 8 reviews and write a summary in his own words.

* Are native English speakers.
* Have at least 1000 completed Amazon tasks (HITs).
* Have 98% approval rate.
* Located in the USA, UK, or Canada.
* Passed a qualification test that assured their understand of the task.


## Annotation Instructions

We used the following instructions:

- The summary should reflect common opinions about the product expressed in the reviews. Try to preserve the common sentiment of the opinions and their details (e.g. what exactly the users like or dislike). For example, if most reviews are negative about the sound quality, then also write negatively about it. Please make the summary coherent and fluent in terms of sentence and information structure. Iterate over the written summary multiple times to improve it, and re-read the reviews whenever necessary.
- Please write your summary as if it were a review itself, e.g. ’This place is expensive’ instead of ’Users thought this place was expensive’. Keep the length of the summary reasonably close to the average length of the reviews.
- Please try to write the summary using your own words instead of copying text directly from the reviews. Using the exact words from the reviews is allowed, but do not copy more than 5 consecutive words from a review.
