import * as d3 from 'd3';

export class Player {
    private timerIndex = 0;
    private isPlaying = false;
    private callback: (isPlaying: boolean) => void = null;
    private stepFunc: (count: number) => void = null;

    onPlayPause(callback: (isPlaying: boolean) => void) {
        this.callback = callback;
    }

    onTick(stepFunc: (count: number) => void) {
        this.stepFunc = stepFunc;
    }

    playOrPause() {
        if (this.isPlaying) {
            this.isPlaying = false;
            this.pause();
        } else {
            this.isPlaying = true;
            this.play();
        }
    }

    play() {
        this.pause();
        this.isPlaying = true;
        if (this.callback) {
            this.callback(this.isPlaying);
        }
        this.start(this.timerIndex);
    }

    pause() {
        this.timerIndex++;
        this.isPlaying = false;
        if (this.callback) {
            this.callback(this.isPlaying);
        }
    }

    private start(localTimerIndex: number) {
        d3.timer(() => {
            if (localTimerIndex < this.timerIndex) {
                return true;  // Done.
            }
            this.stepFunc(1);
            return false;  // Not done.
        }, 0);
    }
}