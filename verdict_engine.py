import math
from typing import List, Dict, Any
from collections import defaultdict

class VerdictEngine:
    def __init__(self):
        pass

    def calculate_unified_score(self, analyzed_reviews: List[Dict[str, Any]]) -> float:
        """
        Calculates a unified sentiment score from 0 to 1 based on all analyzed reviews,
        incorporating the raw star rating and semantic score.
        """
        if not analyzed_reviews:
            return 0.5
            
        total_weight = 0
        weighted_score_sum = 0
        
        for rev in analyzed_reviews:
            semantic_score = rev['analysis']['overall_sentiment']['score']
            star_rating = rev.get('rating', 0)
            
            # Normalize star rating to 0-1
            normalized_star = (star_rating - 1) / 4.0 if star_rating > 0 else semantic_score
            
            # Blend semantic score and star rating (equal weight for accuracy)
            blended_score = (semantic_score * 0.5) + (normalized_star * 0.5)
            
            # Weight by how confident the semantic analysis was and length of review
            confidence = rev['analysis']['overall_sentiment'].get('confidence', 0.5)
            text_length = len(rev.get('text', '').split())
            length_weight = min(1.5, math.log(max(10, text_length)) / 3.0)
            
            weight = confidence * length_weight
            
            weighted_score_sum += blended_score * weight
            total_weight += weight
            
        return weighted_score_sum / total_weight if total_weight > 0 else 0.5

    def extract_pros_cons_and_tags(self, analyzed_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts and ranks top pros and cons across all reviews, and ranks feature tags.
        """
        aspect_sentiments = defaultdict(lambda: {'score_sum': 0, 'count': 0, 'positive_mentions': 0, 'negative_mentions': 0})
        
        for rev in analyzed_reviews:
            for aspect_data in rev['analysis'].get('aspects', []):
                aspect = aspect_data['aspect'].lower()
                score = aspect_data['score']
                weight = aspect_data['weight']
                occurrences = aspect_data.get('occurrences', 1)
                
                stats = aspect_sentiments[aspect]
                stats['score_sum'] += score * weight * occurrences
                stats['count'] += weight * occurrences
                
                if score > 0.6:
                    stats['positive_mentions'] += occurrences
                elif score < 0.4:
                    stats['negative_mentions'] += occurrences
                    
        # Compile ranked list of tags
        ranked_tags = []
        for aspect, stats in aspect_sentiments.items():
            if stats['count'] == 0:
                continue
            avg_score = stats['score_sum'] / stats['count']
            total_mentions = stats['positive_mentions'] + stats['negative_mentions']
            # Ignore aspects mentioned very rarely unless it's a small dataset
            if total_mentions < (len(analyzed_reviews) * 0.05) and len(analyzed_reviews) > 10:
                continue
                
            ranked_tags.append({
                'aspect': aspect.title(),
                'score': avg_score,
                'positive_mentions': stats['positive_mentions'],
                'negative_mentions': stats['negative_mentions'],
                'total_mentions': total_mentions
            })
            
        ranked_tags.sort(key=lambda x: x['total_mentions'], reverse=True)
        
        # Determine Pros and Cons from top tags
        pros = []
        cons = []
        for tag in ranked_tags:
            score = tag['score']
            if score >= 0.65 and len(pros) < 5:
                pros.append(tag)
            elif score <= 0.35 and len(cons) < 5:
                cons.append(tag)
        
        # Order pros by highest score, cons by lowest score
        pros.sort(key=lambda x: x['score'], reverse=True)
        cons.sort(key=lambda x: x['score'])
        
        return {
            'ranked_tags': ranked_tags,
            'pros': pros,
            'cons': cons
        }

    def generate_verdict(self, unified_score: float, pros: List[Dict], cons: List[Dict]) -> str:
        """
        Generates the final human-readable verdict action based on scores.
        """
        if unified_score >= 0.85:
            return "🏆 Excellent — Highly Recommended"
        elif unified_score >= 0.75:
            return "✅ Very Good — Worth Buying"
        elif unified_score >= 0.65:
            return "👍 Good — Solid Choice"
        elif unified_score >= 0.55:
            return "⚖️ Decent — Has Some Drawbacks"
        elif unified_score >= 0.45:
            return "⚠️ Mixed — Proceed with Caution"
        elif unified_score >= 0.35:
            return "👎 Below Average — Wait for a Better Deal"
        else:
            return "🚫 Poor — Best to Avoid"

    def detect_recent_batch_issues(self, analyzed_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Attempts to parse dates and compare the sentiment of the most recent 20%
        of reviews against the historical 80% to detect sudden quality drops.
        """
        import re
        from datetime import datetime
        
        parsed_reviews = []
        for rev in analyzed_reviews:
            date_str = rev.get('date', '')
            score = rev['analysis']['overall_sentiment']['score']
            
            # Simple regex to extract something like "27 July 2022"
            match = re.search(r'(\d+)\s+([a-zA-Z]+)\s+(\d{4})', date_str)
            if match:
                try:
                    date_obj = datetime.strptime(match.group(0), '%d %B %Y')
                    parsed_reviews.append((date_obj, score))
                except:
                    pass
                    
        # Base response includes the trend data as long as we have some parsed reviews
        response = {
            'has_issue': False,
            'message': "Not enough dated reviews to establish a reliable historical trend alert.",
            'trend_data': [{'date': x[0].strftime('%Y-%m-%d'), 'score': x[1]} for x in reversed(parsed_reviews)]
        }
        
        # If we couldn't parse enough dates, skip the alert logic but still return the graph data
        if len(parsed_reviews) < 10:
            return response
            
        # Sort newest to oldest
        parsed_reviews.sort(key=lambda x: x[0], reverse=True)
        
        recent_count = max(5, int(len(parsed_reviews) * 0.25))
        recent_batch = parsed_reviews[:recent_count]
        historical_batch = parsed_reviews[recent_count:]
        
        if not historical_batch:
            return response
            
        recent_avg = sum(x[1] for x in recent_batch) / len(recent_batch)
        hist_avg = sum(x[1] for x in historical_batch) / len(historical_batch)
        
        response['recent_score'] = recent_avg
        response['historical_score'] = hist_avg
        response['message'] = "Recent reviews track closely with the historical average."
        
        # If recent is significantly worse than historical, flag it
        if hist_avg > 0.6 and recent_avg < 0.45:
            response['has_issue'] = True
            response['message'] = "Recent reviews show a significant drop in quality compared to the historical average. This could indicate a bad recent manufacturing batch or firmware update."
            
        return response

    def calculate_platform_scores(self, analyzed_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates split scores for Amazon and Flipkart and checks for alarming discrepancies."""
        amazon_reviews = [r for r in analyzed_reviews if r.get('source', 'Amazon').lower() == 'amazon']
        flipkart_reviews = [r for r in analyzed_reviews if r.get('source', '').lower() == 'flipkart']
        
        amazon_score = self.calculate_unified_score(amazon_reviews) if amazon_reviews else None
        flipkart_score = self.calculate_unified_score(flipkart_reviews) if flipkart_reviews else None
        
        discrepancy_alert = None
        if amazon_score is not None and flipkart_score is not None:
            if abs(amazon_score - flipkart_score) > 0.3:
                discrepancy_alert = f"Significant rating discrepancy detected between platforms! Amazon ({amazon_score:.0%}) vs Flipkart ({flipkart_score:.0%}). This could indicate platform-specific seller issues, counterfeits, or poor shipping."
                
        return {
            'amazon_score': amazon_score,
            'amazon_count': len(amazon_reviews),
            'flipkart_score': flipkart_score,
            'flipkart_count': len(flipkart_reviews),
            'discrepancy_alert': discrepancy_alert
        }

    def aggregate_batch(self, analyzed_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main function to turn a list of analyzed reviews into dashboard-ready stats.
        """
        if not analyzed_reviews:
            return {}
            
        unified_score = self.calculate_unified_score(analyzed_reviews)
        features = self.extract_pros_cons_and_tags(analyzed_reviews)
        verdict = self.generate_verdict(unified_score, features['pros'], features['cons'])
        recent_trends = self.detect_recent_batch_issues(analyzed_reviews)
        platform_stats = self.calculate_platform_scores(analyzed_reviews)
        
        return {
            'unified_score': unified_score,
            'verdict': verdict,
            'pros': features['pros'],
            'cons': features['cons'],
            'ranked_tags': features['ranked_tags'],
            'total_reviews': len(analyzed_reviews),
            'recent_trends': recent_trends,
            'platform_stats': platform_stats
        }

verdict_engine = VerdictEngine()
