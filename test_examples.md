# Test Examples for Granular Sentiment Detection

## Test Case 1: Very Negative (0-10%)
Terrible product. Complete waste of money. Don't buy this garbage. Worst purchase ever!

## Test Case 2: Highly Negative (10-20%)
Very disappointed with this product. It broke after just two days. Poor quality and terrible customer service.

## Test Case 3: Moderately Negative (20-30%)
The product has some decent features but overall I'm disappointed. Battery life is poor and it feels cheap. Not worth the price.

## Test Case 4: Slightly Negative (30-40%)
It's okay but not great. Some features work well, others could be much better. Expected more for this price point.

## Test Case 5: Neutral/Leaning Negative (40-50%)
Average product. Nothing special but it does the basic job. Some good points, some bad points. Depends on what you need.

## Test Case 6: Leaning Positive (50-60%)
Generally satisfied with this purchase. It has a few minor issues but overall it works as advertised. Good value for money.

## Test Case 7: Slightly Positive (60-70%)
Pretty good overall. A few minor issues but generally satisfied. Would recommend with some reservations.

## Test Case 8: Moderately Positive (70-80%)
Really happy with this product! Great features and good build quality. Minor issues but nothing major.

## Test Case 9: Highly Positive (80-90%)
Excellent product! Really impressed with the quality and features. Highly recommend to everyone.

## Test Case 10: Very Positive (90-100%)
Absolutely perfect! Best purchase I've made all year. Outstanding quality, amazing features, flawless performance. Couldn't be happier!

---

## Sarcasm Test Cases

### Sarcasm Test 1
Oh great, another product that breaks after one day. Just what I needed! Perfect.

### Sarcasm Test 2
Yeah right, this is "high quality". Sure it is. Wonderful purchase. Real genius design.

### Irony Test
I absolutely love how it crashes every five minutes. Perfect for getting work done!

---

## Contradiction Test Cases

### Contradiction Test 1
The camera quality is amazing and the screen is gorgeous, but the battery drains way too quickly and it's overpriced. Software is buggy.

### Contradiction Test 2
I love the design and it feels premium, however the performance is terrible. The build quality is excellent but it crashes constantly.

### Contradiction Test 3
Great product overall, although there are some significant issues. The features are fantastic yet the user interface is confusing. Love it despite its flaws.

---

## Mixed Aspect Reviews

### Mixed Review 1
The camera is excellent - takes stunning photos even in low light. The battery life is acceptable, lasting about a day. However, the screen quality is disappointing and the price is way too high for what you get.

### Mixed Review 2
Audio quality is top-notch, really crisp and clear. Design is sleek and modern. But the connectivity issues are frustrating - it keeps disconnecting from Bluetooth. The app is buggy and crashes frequently.

---

## Complex Review (Tests Multiple Features)

The phone has an absolutely amazing camera that takes professional-quality photos. The screen is gorgeous with vibrant colors. Build quality feels premium and solid. However, the battery life is terrible - barely lasts half a day. Oh great, and it comes with bloatware that you can't uninstall. Perfect. The software is buggy and crashes regularly, although the hardware is excellent. Price is way too high for all these issues. I love the design but hate the performance. It's literally the worst and best phone I've ever owned.

Expected Output:
- Overall Sentiment: Mixed/Neutral
- Humor Detected: Sarcasm ("Oh great", "Perfect", "literally")
- Contradictions: Multiple
  - "amazing camera" BUT "battery life is terrible"
  - "gorgeous screen" BUT "software is buggy"
  - "excellent hardware" BUT "hate the performance"
  - "love the design" BUT "worst phone"
- Aspects:
  - Camera: Very Positive (90%+)
  - Screen: Highly Positive (80-90%)
  - Build Quality: Highly Positive (80-90%)
  - Battery: Very Negative (0-10%)
  - Software: Highly Negative (10-20%)
  - Price: Moderately Negative (20-30%)
