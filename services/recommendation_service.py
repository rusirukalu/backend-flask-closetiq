# services/recommendation_service.py - Advanced Outfit Recommendation Engine
import numpy as np
import json
import itertools
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class OutfitRecommendationEngine:
    """Advanced AI-powered outfit recommendation system"""
    
    def __init__(self):
        self.outfit_templates = self.load_outfit_templates()
        self.style_rules = self.load_style_rules()
        self.season_guidelines = self.load_season_guidelines()
        self.occasion_mappings = self.load_occasion_mappings()
        
        # Performance tracking
        self.generated_outfits = 0
        self.successful_recommendations = 0
    
    def load_outfit_templates(self):
        """Load outfit templates for different occasions"""
        return {
            'work': {
                'required': ['tops', 'bottoms', 'shoes'],
                'optional': ['outerwear', 'accessories'],
                'formality_min': 7,
                'color_palette': ['neutral', 'conservative'],
                'avoid_patterns': ['flashy', 'overly_casual']
            },
            'casual': {
                'required': ['tops', 'bottoms', 'shoes'],
                'optional': ['accessories', 'outerwear'],
                'formality_min': 3,
                'color_palette': ['any'],
                'patterns_allowed': ['all']
            },
            'formal': {
                'required': ['dress_or_suit', 'shoes'],
                'optional': ['accessories', 'outerwear'],
                'formality_min': 8,
                'color_palette': ['sophisticated', 'neutral'],
                'fabric_preference': ['silk', 'wool', 'high_quality']
            },
            'party': {
                'required': ['statement_piece', 'shoes'],
                'optional': ['accessories'],
                'formality_min': 6,
                'color_palette': ['bold', 'trendy'],
                'patterns_allowed': ['all']
            },
            'date': {
                'required': ['attractive_top', 'bottoms', 'shoes'],
                'optional': ['accessories'],
                'formality_min': 6,
                'style_preference': ['romantic', 'elegant', 'trendy']
            }
        }
    
    def load_style_rules(self):
        """Load comprehensive style rules"""
        return {
            'color_harmony': {
                'complementary': [('red', 'green'), ('blue', 'orange'), ('yellow', 'purple')],
                'analogous': [('blue', 'green'), ('red', 'orange'), ('yellow', 'green')],
                'monochromatic': ['single_color_family'],
                'neutral_safe': ['black', 'white', 'gray', 'navy', 'beige']
            },
            'proportion_rules': {
                'fitted_loose_balance': 'If top is fitted, bottom should be loose and vice versa',
                'color_distribution': '60% dominant, 30% secondary, 10% accent',
                'pattern_mixing': 'Maximum 2 patterns, different scales'
            },
            'style_coherence': {
                'consistent_formality': 'All pieces should match occasion formality',
                'cohesive_aesthetic': 'Maintain consistent style theme',
                'seasonal_appropriateness': 'Match clothing weight to season'
            }
        }
    
    def load_season_guidelines(self):
        """Load seasonal fashion guidelines"""
        return {
            'spring': {
                'colors': ['pastels', 'light_colors', 'fresh_greens'],
                'fabrics': ['light_cotton', 'linen', 'light_knits'],
                'layering': 'light_layers_for_transition',
                'footwear': ['sneakers', 'light_boots', 'flats']
            },
            'summer': {
                'colors': ['bright', 'white', 'light_colors'],
                'fabrics': ['cotton', 'linen', 'breathable'],
                'silhouettes': ['loose', 'flowing', 'minimal'],
                'footwear': ['sandals', 'canvas_shoes', 'breathable']
            },
            'fall': {
                'colors': ['earth_tones', 'deep_colors', 'warm_hues'],
                'fabrics': ['wool', 'cashmere', 'medium_weight'],
                'layering': 'strategic_layering',
                'footwear': ['boots', 'closed_shoes']
            },
            'winter': {
                'colors': ['rich_colors', 'deep_tones', 'jewel_tones'],
                'fabrics': ['wool', 'cashmere', 'heavy_knits'],
                'layering': 'warmth_priority',
                'footwear': ['warm_boots', 'weather_appropriate']
            }
        }
    
    def load_occasion_mappings(self):
        """Load occasion-specific mappings"""
        return {
            'work': {
                'style_keywords': ['professional', 'polished', 'conservative'],
                'avoid_keywords': ['revealing', 'casual', 'flashy'],
                'preferred_colors': ['navy', 'black', 'gray', 'white', 'burgundy']
            },
            'party': {
                'style_keywords': ['festive', 'eye-catching', 'trendy'],
                'encourage_keywords': ['statement', 'bold', 'glamorous'],
                'preferred_colors': ['jewel_tones', 'metallics', 'bold_colors']
            },
            'casual': {
                'style_keywords': ['comfortable', 'relaxed', 'everyday'],
                'flexibility': 'high',
                'preferred_colors': ['any']
            }
        }
    
    def generate_outfits(self, user_id, wardrobe_items, occasion='casual', season='spring', 
                        weather_context=None, style_preferences=None, count=5):
        """Generate AI-powered outfit recommendations"""
        try:
            start_time = datetime.now()
            
            if not wardrobe_items:
                return {
                    'outfits': [],
                    'total': 0,
                    'algorithm_info': {'error': 'No wardrobe items provided'}
                }
            
            # Categorize wardrobe items
            categorized_items = self.categorize_wardrobe_items(wardrobe_items)
            
            # Get outfit template for occasion
            template = self.outfit_templates.get(occasion, self.outfit_templates['casual'])
            
            # Generate outfit combinations
            outfits = self.create_outfit_combinations(
                categorized_items, template, occasion, season, weather_context, style_preferences
            )
            
            # Score and rank outfits
            scored_outfits = self.score_outfits(outfits, occasion, season, weather_context)
            
            # Select top outfits
            top_outfits = sorted(scored_outfits, key=lambda x: x['overall_score'], reverse=True)[:count]
            
            # Generate explanations
            explained_outfits = self.add_explanations(top_outfits, occasion, season)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update performance metrics
            self.generated_outfits += len(outfits)
            self.successful_recommendations += len(explained_outfits)
            
            return {
                'outfits': explained_outfits,
                'total': len(explained_outfits),
                'algorithm_info': {
                    'total_combinations_considered': len(outfits),
                    'processing_time_ms': processing_time,
                    'occasion': occasion,
                    'season': season,
                    'template_used': template,
                    'weather_considered': weather_context is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Outfit generation error: {str(e)}")
            raise
    
    def categorize_wardrobe_items(self, items):
        """Categorize wardrobe items by type"""
        categories = {
            'tops': [],
            'bottoms': [],
            'dresses': [],
            'shoes': [],
            'outerwear': [],
            'accessories': []
        }
        
        # Category mapping
        category_mapping = {
            'shirts_blouses': 'tops',
            'tshirts_tops': 'tops',
            'sweaters': 'tops',
            'dresses': 'dresses',
            'pants_jeans': 'bottoms',
            'shorts': 'bottoms',
            'skirts': 'bottoms',
            'shoes_sneakers': 'shoes',
            'shoes_formal': 'shoes',
            'jackets_coats': 'outerwear',
            'bags_accessories': 'accessories'
        }
        
        for item in items:
            # Assume item has category information
            item_category = item.get('category', 'unknown')
            mapped_category = category_mapping.get(item_category, 'accessories')
            
            if mapped_category in categories:
                categories[mapped_category].append(item)
        
        return categories
    
    def create_outfit_combinations(self, categorized_items, template, occasion, season, weather_context, style_preferences):
        """Create outfit combinations based on template and items"""
        outfits = []
        
        # Get required categories from template
        required_categories = template['required']
        optional_categories = template.get('optional', [])
        
        # Map template categories to actual categories
        category_mapping = {
            'tops': 'tops',
            'bottoms': 'bottoms',
            'dress_or_suit': 'dresses',
            'attractive_top': 'tops',
            'statement_piece': ['tops', 'dresses'],
            'shoes': 'shoes',
            'outerwear': 'outerwear',
            'accessories': 'accessories'
        }
        
        # Check if we have required items
        available_for_required = {}
        for req_cat in required_categories:
            mapped_cats = category_mapping.get(req_cat, req_cat)
            if isinstance(mapped_cats, list):
                # Multiple categories can fulfill this requirement
                items = []
                for cat in mapped_cats:
                    items.extend(categorized_items.get(cat, []))
                available_for_required[req_cat] = items
            else:
                available_for_required[req_cat] = categorized_items.get(mapped_cats, [])
        
        # Generate combinations
        if all(len(items) > 0 for items in available_for_required.values()):
            # Create combinations of required items
            required_combinations = list(itertools.product(*available_for_required.values()))
            
            for combo in required_combinations[:50]:  # Limit combinations for performance
                outfit = {
                    'items': list(combo),
                    'item_ids': [item.get('_id', f'item_{i}') for i, item in enumerate(combo)],
                    'categories_covered': required_categories.copy(),
                    'occasion': occasion,
                    'season': season
                }
                
                # Add optional items if available and beneficial
                for opt_cat in optional_categories:
                    mapped_cat = category_mapping.get(opt_cat, opt_cat)
                    available_optional = categorized_items.get(mapped_cat, [])
                    
                    if available_optional and len(outfit['items']) < 5:  # Don't overcomplicate
                        # Select best optional item based on compatibility
                        best_optional = self.select_best_optional_item(outfit['items'], available_optional)
                        if best_optional:
                            outfit['items'].append(best_optional)
                            outfit['item_ids'].append(best_optional.get('_id', f'opt_item_{len(outfit["items"])}'))
                            outfit['categories_covered'].append(opt_cat)
                
                outfits.append(outfit)
        
        return outfits
    
    def select_best_optional_item(self, existing_items, optional_items):
        """Select the best optional item to complement existing outfit"""
        if not optional_items:
            return None
        
        best_item = None
        best_score = 0
        
        for item in optional_items:
            # Calculate compatibility with existing items
            compatibility_score = self.calculate_item_compatibility(item, existing_items)
            
            if compatibility_score > best_score:
                best_score = compatibility_score
                best_item = item
        
        # Only add if compatibility is good enough
        return best_item if best_score > 0.6 else None
    
    def calculate_item_compatibility(self, item, existing_items):
        """Calculate how well an item fits with existing outfit items"""
        if not existing_items:
            return 0.5
        
        # Simplified compatibility calculation
        # In a full implementation, this would use the compatibility engine
        
        item_colors = item.get('attributes', {}).get('colors', [])
        item_style = item.get('attributes', {}).get('style', ['neutral'])
        
        compatibility_scores = []
        
        for existing_item in existing_items:
            existing_colors = existing_item.get('attributes', {}).get('colors', [])
            existing_style = existing_item.get('attributes', {}).get('style', ['neutral'])
            
            # Color compatibility (simplified)
            color_score = 0.7  # Default neutral
            if any(color in existing_colors for color in item_colors):
                color_score = 0.9  # Same color family
            elif any(color in ['black', 'white', 'gray'] for color in item_colors + existing_colors):
                color_score = 0.8  # Neutral colors
            
            # Style compatibility (simplified)
            style_score = 0.7  # Default neutral
            if any(style in existing_style for style in item_style):
                style_score = 0.9  # Same style
            
            overall_score = (color_score + style_score) / 2
            compatibility_scores.append(overall_score)
        
        return np.mean(compatibility_scores) if compatibility_scores else 0.5
    
    def score_outfits(self, outfits, occasion, season, weather_context):
        """Score and rank outfit combinations"""
        scored_outfits = []
        
        for outfit in outfits:
            scores = {
                'color_harmony': self.score_color_harmony(outfit),
                'style_coherence': self.score_style_coherence(outfit),
                'occasion_appropriateness': self.score_occasion_appropriateness(outfit, occasion),
                'seasonal_appropriateness': self.score_seasonal_appropriateness(outfit, season),
                'weather_appropriateness': self.score_weather_appropriateness(outfit, weather_context),
                'completeness': self.score_outfit_completeness(outfit),
                'novelty': self.score_outfit_novelty(outfit)
            }
            
            # Calculate weighted overall score
            weights = {
                'color_harmony': 0.20,
                'style_coherence': 0.20,
                'occasion_appropriateness': 0.25,
                'seasonal_appropriateness': 0.15,
                'weather_appropriateness': 0.10,
                'completeness': 0.05,
                'novelty': 0.05
            }
            
            overall_score = sum(scores[metric] * weights[metric] for metric in scores)
            
            outfit['scores'] = scores
            outfit['overall_score'] = overall_score
            outfit['grade'] = self.calculate_grade(overall_score)
            
            scored_outfits.append(outfit)
        
        return scored_outfits
    
    def score_color_harmony(self, outfit):
        """Score color harmony in the outfit"""
        colors = []
        for item in outfit['items']:
            item_colors = item.get('attributes', {}).get('colors', [])
            colors.extend(item_colors)
        
        if len(colors) <= 1:
            return 0.8  # Single color is safe
        
        # Check for neutral dominance
        neutrals = ['black', 'white', 'gray', 'beige', 'navy']
        neutral_count = sum(1 for color in colors if color in neutrals)
        
        if neutral_count >= len(colors) * 0.7:
            return 0.9  # Neutral palette is safe
        
        # Check for complementary colors (simplified)
        complementary_pairs = [('red', 'green'), ('blue', 'orange'), ('yellow', 'purple')]
        for color1, color2 in complementary_pairs:
            if color1 in colors and color2 in colors:
                return 0.95  # Complementary colors work well
        
        # Default scoring based on color count
        if len(set(colors)) <= 3:
            return 0.8  # Good color variety
        elif len(set(colors)) <= 5:
            return 0.6  # Acceptable variety
        else:
            return 0.4  # Too many colors
    
    def score_style_coherence(self, outfit):
        """Score style coherence across outfit items"""
        styles = []
        for item in outfit['items']:
            item_styles = item.get('attributes', {}).get('style', ['neutral'])
            styles.extend(item_styles)
        
        if not styles:
            return 0.5
        
        # Count unique styles
        unique_styles = set(styles)
        
        if len(unique_styles) == 1:
            return 0.95  # Perfect coherence
        elif len(unique_styles) == 2:
            # Check if styles are compatible
            compatible_pairs = [('classic', 'elegant'), ('casual', 'comfortable'), ('formal', 'professional')]
            for style1, style2 in compatible_pairs:
                if style1 in unique_styles and style2 in unique_styles:
                    return 0.9  # Compatible styles
            return 0.7  # Moderately compatible
        elif len(unique_styles) <= 3:
            return 0.6  # Some coherence
        else:
            return 0.3  # Poor coherence
    
    def score_occasion_appropriateness(self, outfit, occasion):
        """Score how appropriate the outfit is for the occasion"""
        template = self.outfit_templates.get(occasion, self.outfit_templates['casual'])
        
        # Check formality level
        formality_scores = []
        for item in outfit['items']:
            # Simplified formality scoring
            category = item.get('category', 'unknown')
            if category in ['shirts_blouses', 'jackets_coats', 'shoes_formal']:
                formality_scores.append(8)  # High formality
            elif category in ['dresses']:
                formality_scores.append(7)  # Medium-high formality
            elif category in ['sweaters', 'pants_jeans']:
                formality_scores.append(5)  # Medium formality
            else:
                formality_scores.append(3)  # Low formality
        
        avg_formality = np.mean(formality_scores) if formality_scores else 5
        required_formality = template.get('formality_min', 5)
        
        # Score based on formality match
        if avg_formality >= required_formality:
            return min(1.0, avg_formality / 10)  # Good match
        else:
            return max(0.3, avg_formality / required_formality)  # Penalize under-dressing
    
    def score_seasonal_appropriateness(self, outfit, season):
        """Score seasonal appropriateness"""
        season_guidelines = self.season_guidelines.get(season, {})
        
        # Check fabric appropriateness (simplified)
        appropriate_fabrics = season_guidelines.get('fabrics', [])
        colors = season_guidelines.get('colors', [])
        
        score = 0.7  # Base score
        
        # Bonus for seasonal colors
        outfit_colors = []
        for item in outfit['items']:
            outfit_colors.extend(item.get('attributes', {}).get('colors', []))
        
        if any(color in colors for color in outfit_colors):
            score += 0.2
        
        # Check for season-inappropriate items
        if season == 'summer':
            heavy_items = ['jackets_coats', 'sweaters']
            if any(item.get('category') in heavy_items for item in outfit['items']):
                score -= 0.3
        elif season == 'winter':
            light_items = ['shorts', 'sandals']
            if any(item.get('category') in light_items for item in outfit['items']):
                score -= 0.3
        
        return max(0.1, min(1.0, score))
    
    def score_weather_appropriateness(self, outfit, weather_context):
        """Score weather appropriateness"""
        if not weather_context:
            return 0.7  # Neutral if no weather data
        
        temperature = weather_context.get('temperature', 20)
        conditions = weather_context.get('conditions', '').lower()
        
        score = 0.7  # Base score
        
        # Temperature appropriateness
        if temperature < 10:  # Cold
            if any(item.get('category') in ['jackets_coats', 'sweaters'] for item in outfit['items']):
                score += 0.2
            if any(item.get('category') in ['shorts', 'sandals'] for item in outfit['items']):
                score -= 0.3
        elif temperature > 25:  # Hot
            if any(item.get('category') in ['shorts', 'tshirts_tops'] for item in outfit['items']):
                score += 0.2
            if any(item.get('category') in ['jackets_coats', 'sweaters'] for item in outfit['items']):
                score -= 0.3
        
        # Rain appropriateness
        if 'rain' in conditions:
            # Should have appropriate footwear and outerwear
            if any(item.get('category') in ['jackets_coats'] for item in outfit['items']):
                score += 0.1
        
        return max(0.1, min(1.0, score))
    
    def score_outfit_completeness(self, outfit):
        """Score how complete the outfit is"""
        required_categories = ['tops', 'bottoms', 'shoes']
        covered_categories = outfit.get('categories_covered', [])
        
        # Map categories
        category_map = {
            'tops': ['tops', 'attractive_top'],
            'bottoms': ['bottoms'],
            'shoes': ['shoes'],
            'dress_or_suit': ['dresses']  # Dresses can replace tops+bottoms
        }
        
        covered_mapped = set()
        for cat in covered_categories:
            if cat in category_map:
                covered_mapped.update(category_map[cat])
            else:
                covered_mapped.add(cat)
        
        # Special case: dress covers tops+bottoms
        if 'dresses' in covered_mapped:
            covered_mapped.update(['tops', 'bottoms'])
        
        missing_required = set(required_categories) - covered_mapped
        
        if not missing_required:
            return 1.0  # Complete outfit
        else:
            return max(0.3, 1.0 - len(missing_required) * 0.3)  # Penalize missing pieces
    
    def score_outfit_novelty(self, outfit):
        """Score outfit novelty/creativity"""
        # This is a simplified novelty score
        # In a full implementation, this would check against user's outfit history
        
        unique_categories = set(item.get('category') for item in outfit['items'])
        
        if len(unique_categories) >= 4:
            return 0.9  # Good variety
        elif len(unique_categories) == 3:
            return 0.7  # Standard variety
        else:
            return 0.5  # Limited variety
    
    def calculate_grade(self, score):
        """Calculate letter grade from numerical score"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.85:
            return 'A'
        elif score >= 0.8:
            return 'A-'
        elif score >= 0.75:
            return 'B+'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.65:
            return 'B-'
        elif score >= 0.6:
            return 'C+'
        elif score >= 0.55:
            return 'C'
        elif score >= 0.5:
            return 'C-'
        else:
            return 'D'
    
    def add_explanations(self, outfits, occasion, season):
        """Add explanations for outfit recommendations"""
        for outfit in outfits:
            explanations = []
            
            # Color explanation
            if outfit['scores']['color_harmony'] > 0.8:
                explanations.append("Colors work harmoniously together")
            
            # Style explanation
            if outfit['scores']['style_coherence'] > 0.8:
                explanations.append("Consistent style theme throughout")
            
            # Occasion explanation
            if outfit['scores']['occasion_appropriateness'] > 0.8:
                explanations.append(f"Perfect for {occasion} occasions")
            
            # Season explanation
            if outfit['scores']['seasonal_appropriateness'] > 0.8:
                explanations.append(f"Seasonally appropriate for {season}")
            
            outfit['explanation'] = explanations
            outfit['recommendation_reason'] = f"This outfit scores {outfit['grade']} with strong {', '.join(explanations[:2])}"
        
        return outfits
    
    def get_performance_metrics(self):
        """Get recommendation engine performance metrics"""
        success_rate = (
            self.successful_recommendations / max(self.generated_outfits, 1) * 100
        )
        
        return {
            'total_outfits_generated': self.generated_outfits,
            'successful_recommendations': self.successful_recommendations,
            'success_rate_percentage': success_rate,
            'outfit_templates_loaded': len(self.outfit_templates),
            'style_rules_loaded': len(self.style_rules),
            'season_guidelines_loaded': len(self.season_guidelines)
        }
