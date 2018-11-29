class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        
        self.fitx = None
        
        self.past_good_fit = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #average x values of the fitted line over the last n iterations
        self.bestx = None 
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
