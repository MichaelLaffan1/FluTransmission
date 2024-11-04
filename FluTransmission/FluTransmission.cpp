#include <iostream>
#include <fstream>
#include <omp.h>

using namespace std;

int gridHeight;
int gridWidth;
double alpha;
double beta;
int omega;
int numDays;
int numThreads;

class Person {
public:
    bool was_infected;
    int sick_days;

    Person() : was_infected(false), sick_days(0) {}
};

void readSettingsFromFile(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open settings file.\n";
        return;
    }

    char label[50];
    if (file >> label >> gridHeight &&
        file >> label >> gridWidth &&
        file >> label >> alpha &&
        file >> label >> beta &&
        file >> label >> omega &&
        file >> label >> numDays) {
        numThreads = 5;
    }
    else {
        std::cerr << "Error reading settings file.\n";
    }

    file.close();
}

unsigned int threadRand() {
    static thread_local unsigned int seed = 123456789;
    seed = seed * 1103515245 + 12345;
    return (seed / 65536) % 32768;
}

void initializeGrid(Person** grid, double alpha) {
    int totalPeople = gridHeight * gridWidth;
    int infectedCount = static_cast<int>(alpha * totalPeople);
    int placedInfected = 0;
    while (placedInfected < infectedCount) {
        int i = threadRand() % gridHeight;
        int j = threadRand() % gridWidth;

        if (!grid[i][j].was_infected) {
            grid[i][j].was_infected = true;
            grid[i][j].sick_days = 1;
            placedInfected++;
        }
    }
}

void printGridToFile(Person** grid, int day) {
    std::ofstream file("flu_simulation.txt", std::ios::app);
    file << "Day " << day << ":\n";
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            file << (grid[i][j].sick_days > 0 ? 1 : 0) << " ";
        }
        file << "\n";
    }
    file.close();
}

void printDebugGridToFile(int** debugGrid, int day) {
    std::ofstream file("flu_debug.txt", std::ios::app);
    file << "Day " << day << " (Thread IDs):\n";
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            file << debugGrid[i][j] << " ";
        }
        file << "\n";
    }
    file.close();
}

void updateGrid(Person** grid, Person** newGrid, int** debugGrid) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            Person& current = grid[i][j];
            cout << omp_get_thread_num();
            debugGrid[i][j] = omp_get_thread_num();

            // Copy and update current state for sick people
            if (current.sick_days > 0) {
                newGrid[i][j] = current;
                newGrid[i][j].sick_days++;
                if (newGrid[i][j].sick_days >= omega) {
                    newGrid[i][j].sick_days = 0;
                }
            }
            else {
                newGrid[i][j] = current;
            }
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < gridHeight; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            if (newGrid[i][j].sick_days == 0 && !newGrid[i][j].was_infected) {
                int sickNeighbors = 0;
                if (i > 0 && grid[i - 1][j].sick_days > 0) sickNeighbors++;
                if (i < gridHeight - 1 && grid[i + 1][j].sick_days > 0) sickNeighbors++;
                if (j > 0 && grid[i][j - 1].sick_days > 0) sickNeighbors++;
                if (j < gridWidth - 1 && grid[i][j + 1].sick_days > 0) sickNeighbors++;

                int randVal = threadRand() % 1000;
                if (randVal < beta * sickNeighbors * 1000) {
                    newGrid[i][j].was_infected = true;
                    newGrid[i][j].sick_days = 1;
                }
            }
        }
    }
}

int main() {
    readSettingsFromFile("settings.txt");
    omp_set_num_threads(numThreads);

    std::ofstream file("flu_simulation.txt", std::ios::trunc);
    file.close();

    Person** grid = new Person * [gridHeight];
    Person** newGrid = new Person * [gridHeight];
    for (int i = 0; i < gridHeight; ++i) {
        grid[i] = new Person[gridWidth];
        newGrid[i] = new Person[gridWidth];
    }

    int** debugGrid = new int* [gridHeight];
    for (int i = 0; i < gridHeight; ++i) {
        debugGrid[i] = new int[gridWidth];
        std::fill(debugGrid[i], debugGrid[i] + gridWidth, -1);
    }

    initializeGrid(grid, alpha);
    printGridToFile(grid, 0);
    printDebugGridToFile(debugGrid, 0);

    for (int day = 1; day <= numDays; ++day) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < gridHeight; ++i) {
            for (int j = 0; j < gridWidth; ++j) {
                newGrid[i][j] = Person();
            }
        }

        updateGrid(grid, newGrid, debugGrid);
        printGridToFile(newGrid, day);
        printDebugGridToFile(debugGrid, day);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < gridHeight; ++i) {
            for (int j = 0; j < gridWidth; ++j) {
                grid[i][j] = newGrid[i][j];
            }
        }
    }

    for (int i = 0; i < gridHeight; ++i) {
        delete[] grid[i];
        delete[] newGrid[i];
        delete[] debugGrid[i];
    }
    delete[] grid;
    delete[] newGrid;
    delete[] debugGrid;

    return 0;
}
