## This Makefile can be used to compile the serial version of the
## K-Means program. The following targets are available:
##
## ALL        compile k-means and inputgen
## demo       compile k-means, run it on the demo dataset and produce demo.mp4
## clean      remove temporary files
##
## Last modified on 2025-12-03 by Moreno Marzolla

CFLAGS+=-std=c99 -Wall -Wpedantic
EXES:=k-means inputgen

.PHONY: demo clean

ALL: $(EXES)

demo:
	$(CC) $(CFLAGS) -DMAKE_MOVIE k-means.c -o k-means
	\rm -f centroids_*.txt clusters_*.txt img_*.png
	./k-means 5 demo.in demo.out
	\rm demo.out
	./generate_frames.sh
	ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "img_*.png" -vcodec mpeg4 -r 1 demo.mp4

demo.in: inputgen
	./inputgen 20 2 50 > $@

clean:
	\rm -f $(EXES) *.o *~ centroids_*.txt clusters_*.txt img_*.png demo.out demo.mp4
