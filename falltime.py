import argparse
import numpy as np

def falltime(h,g):
    # h is height, g is gravity
    #calculates the time it takes to fall from a given height under gravity
    time_taken = np.sqrt((2 * h) / g)
    final_velocity = g * time_taken
    
    return float(time_taken), float(final_velocity)



def main():
    parser = argparse.ArgumentParser(description="calculate the time it takes for a ball to drop to the ground")
    
    parser.add_argument("-H", "--height", type=float, required=True, help="height from where the ball is dropped (in meters, feet, etc.)")
    #adding an argparse thing to call and set height
    
    parser.add_argument("-g", "--gravity", type=float, default=9.81, help="acceleration due to gravity (default: 9.81 m/s^2, use 32.2 for ft/s^2)")
    #adding an argparse thing to call and set gravity

    #adding something if an unacceptable value is inputted
    args = parser.parse_args()
    
    if args.height < 0:
        print("Error: Height cannot be negative!")
        return
        
    if args.gravity <= 0:
        print("Error: Gravity must be a positive value!")
        return
        
    time_taken, final_velocity = falltime(args.height, args.gravity)
    
    print(f"height: {args.height}")
    print(f"gravity: {args.gravity}")
    print(f"time to reach the ground: {time_taken:.4f} seconds")
    print(f"final velocity upon impact: {final_velocity:.4f} units/sec")

if __name__ == "__main__":
    main()
 
