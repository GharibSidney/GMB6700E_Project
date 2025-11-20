import cv2
def draw_charuco_colored(img, charuco_corners, charuco_ids, board):
    img_out = img.copy()

    # Extract chessboard geometry
    squares_x = board.getChessboardSize()[0]
    squares_y = board.getChessboardSize()[1]

    # Convert to simple list
    pts = [tuple(map(int, c.ravel())) for c in charuco_corners]

    # Assign distinct colors (repeat if needed)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    # Draw circles
    for i, (x, y) in enumerate(pts):
        color = colors[i % len(colors)]
        cv2.circle(img_out, (x, y), 6, color, -1)

    # --- Draw grid lines between adjacent valid corners ---
    # ChArUco corner IDs correspond to chessboard intersections
    # id_to_pt = {int(cid): tuple(map(int, c.ravel())) 
    #             for cid, c in zip(charuco_ids, charuco_corners)}

    # # For each node in the grid, connect to right & down neighbors
    # for r in range(squares_y):
    #     for c in range(squares_x):
    #         current_id = r * squares_x + c
    #         right_id = r * squares_x + (c + 1)
    #         down_id = (r + 1) * squares_x + c

    #         # Draw right edge if both corners detected
    #         if current_id in id_to_pt and right_id in id_to_pt:
    #             cv2.line(img_out, id_to_pt[current_id], id_to_pt[right_id], (0, 255, 0), 2)

    #         # Draw down edge if both corners detected
    #         if current_id in id_to_pt and down_id in id_to_pt:
    #             cv2.line(img_out, id_to_pt[current_id], id_to_pt[down_id], (0, 255, 0), 2)

    return img_out
