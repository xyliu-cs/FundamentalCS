#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <ctype.h>

#define ROW 10
#define COLUMN 50
#define LOGWIDTH 15

pthread_mutex_t print_lock;
pthread_mutex_t frog_lock;

int GAME_STATUS = 1;
int MOVE = 0;
int bound_flag = 0;
int water_flag = 0;
int win_flag = 0;
int quit_flag = 0;


struct Node {
	int x, y;
	Node(int _x, int _y) : x(_x), y(_y){};
	Node(){};
} frog;

char map[ROW + 10][COLUMN];
int head_index[ROW];

/* Determine a keyboard is hit or not. If yes, return 1. If not, return 0. */ 
int kbhit(void)
{
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if (ch != EOF) {
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

/*  Check keyboard hits, to change frog's position or quit the game. */
void *user_move(void *t)
{
	char move;
	while (GAME_STATUS) {
		if (kbhit()) {
			move = tolower(getchar());
			usleep(100000);
			switch (move) {
			case 'w':
				MOVE = 1;
				pthread_mutex_lock(&frog_lock);
				if ((frog.x - 1) >= 0)
					frog.x -= 1;
				pthread_mutex_unlock(&frog_lock);
				break;
			case 's':
				MOVE = 2;
				pthread_mutex_lock(&frog_lock);
				if ((frog.x + 1) <= ROW)
					frog.x += 1;
				pthread_mutex_unlock(&frog_lock);
				break;
			case 'a':
				// 3 for left shift
				MOVE = 3;
				pthread_mutex_lock(&frog_lock);
				if (frog.y - 1 >= 0)
					frog.y -= 1;	
				pthread_mutex_unlock(&frog_lock);
				break;
			case 'd':
				MOVE = 4;
				pthread_mutex_lock(&frog_lock);
				if (frog.y + 1 <= COLUMN-2)
					frog.y += 1;
				pthread_mutex_unlock(&frog_lock);
				break;
			case 'q':
				quit_flag = 1;
				GAME_STATUS = 0;
				break;
			}
		}
	}
	pthread_exit(NULL);
}

/*  Print the map on the screen  */
void *display_console(void *)
{
	while (GAME_STATUS) {
		pthread_mutex_lock(&print_lock);
		usleep(100000);
		printf("\033[H\033[J");
		// printf("demo\n");
		for (int i = 0; i <= ROW; i++)
			puts(map[i]);

		pthread_mutex_unlock(&print_lock);
		usleep(20);
	}

	/*  Display the output for user: win, lose or quit.  */
	if (win_flag) {
		usleep(200000);
		printf("\033[H\033[J");
		printf("You win the game!\n");
	}
	else if (bound_flag || water_flag)
	{
		usleep(200000);
		printf("\033[H\033[J");
		printf("You lose the game!\n");
	}
	else if (quit_flag)
	{
		usleep(200000);
		printf("\033[H\033[J");
		printf("You exit the game!\n");
	}
	pthread_exit(NULL);
}

/*  Move the logs and the frog */
void *logs_move(void *t)
{
	/* Logs have the index from 1 to 9 */
	int step_size = 1;
	int hdpos;
	int new_hdpos;


	/*  Check game's status  */
	while (GAME_STATUS) {
		if (bound_flag || water_flag || win_flag) {
			GAME_STATUS = 0;
			break;
		}

		// clear map
		pthread_mutex_lock(&print_lock);
		for (int i = 1; i < ROW; ++i) {
			for (int j = 0; j < COLUMN - 1; ++j)
				map[i][j] = ' ';
		}

		for (int i = 1; i < ROW; i++) {
			// stamp here
			hdpos = head_index[i];
			// must already jumped (console index is 1-9)

			if (i % 2 == 0) {
				// right shift
				new_hdpos = (hdpos + step_size) % (COLUMN - 1);
			} else {
				new_hdpos = (hdpos - step_size);
				if (new_hdpos < 0)
					new_hdpos = COLUMN - 2 +
						    (hdpos - step_size) + 1;
			}
			for (int k = new_hdpos; k < new_hdpos + LOGWIDTH; k++)
				map[i][k % (COLUMN - 1)] = '=';

			// update head index
			head_index[i] = new_hdpos;
		}

		if (MOVE) {
			// tolerate this time
			if ((frog.y == 0 || frog.y == COLUMN - 2) && (frog.x != ROW)) {
				bound_flag = 1;
			}						
			// on board
			if (map[frog.x][frog.y] == '=') {
				map[frog.x][frog.y] = '0';
				// jump from bank, have some minor problem
				if (MOVE == 1 && frog.x == (ROW-1)) {
					map[ROW][frog.y] = '|'; 
				}
				pthread_mutex_lock(&frog_lock);
				// right shift
				if (frog.x % 2 == 0) {
					// right boundary
					if ((frog.y + step_size) > (COLUMN - 2)) {
						frog.y = COLUMN - 2;
					} else {
						frog.y = frog.y + step_size;
					}
				}
				// left shift
				else {
					// left boundary
					if (frog.y - step_size < 0) {
						frog.y = 0;
					} else {
						frog.y = frog.y - step_size;
					}
				}
				pthread_mutex_unlock(&frog_lock);
			}
			// water
			else if (map[frog.x][frog.y] == ' ') {
				map[frog.x][frog.y] = '0';
				if (MOVE == 1 && frog.x == (ROW-1)) {
					map[ROW][frog.y] = '|'; 
				}
				water_flag = 1;
			}
			else if (map[frog.x][frog.y] == '|')
			{
				map[frog.x][frog.y] = '0';
				// start river bank
				if (frog.x == 0) {
					win_flag = 1;
				}
				// frog left shifts
				else if (MOVE == 3) {
					map[frog.x][frog.y+1] = '|';
				}
				// frog right shifts
				else if (MOVE == 4) {
					map[frog.x][frog.y-1] = '|';
				}
			}
			
		}

		// display parameter
		usleep(2000);
		pthread_mutex_unlock(&print_lock);
		usleep(200);

	}

	pthread_exit(NULL);

}

/* Initialize the river map, logs and frog's starting position*/
void initialize_console(void)
{

	memset(map, 0, sizeof(map));
	srand(time(0));

	int i, j, k;
	int head;

	// river bank
	for (j = 0; j < COLUMN - 1; ++j)
		map[ROW][j] = map[0][j] = '|';

	for (j = 0; j < COLUMN - 1; ++j)
		map[0][j] = map[0][j] = '|';

	for (i = 1; i < ROW; ++i) {
		for (j = 0; j < COLUMN - 1; ++j)
			map[i][j] = ' ';
	}

	// frog
	frog = Node(ROW, (COLUMN - 1) / 2);
	map[frog.x][frog.y] = '0';

	// log
	for (i = 1; i < ROW; ++i) {
		head = (rand() % (COLUMN - 1));
		head_index[i] = head;
		for (j = head; j < head + LOGWIDTH; j++)
			map[i][j % (COLUMN - 1)] = '=';
	}

	return;
}


int main(int argc, char *argv[])
{
	int i, j, k;

	initialize_console();

	/*  Create pthreads for wood move and frog control.  */
	pthread_t capture_t, move_t, display_t;
	pthread_mutex_init(&print_lock, NULL);
	pthread_mutex_init(&frog_lock, NULL);

	// hide cursor
	printf("\e[?25l");

	i = pthread_create(&capture_t, NULL, user_move, NULL);
	if (i != 0) {
		printf("create user_move thread failed, %d\n", i);
		exit(1);
	}
	j = pthread_create(&move_t, NULL, logs_move, NULL);
	if (j != 0) {
		printf("create logs_move thread failed, %d\n", i);
		exit(1);
	}
	k = pthread_create(&display_t, NULL, display_console, NULL);
	if (k != 0) {
		printf("create display_console thread failed, %d\n", i);
		exit(1);
	}

	pthread_join(capture_t, NULL);
	pthread_join(move_t, NULL);
	pthread_join(display_t, NULL);


	pthread_mutex_destroy(&print_lock);
	pthread_mutex_destroy(&frog_lock);

	// display cursor
	printf("\e[?25h");
	pthread_exit(NULL);


	return 0;
}
