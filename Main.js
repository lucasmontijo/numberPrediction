const tf = require('@tensorflow/tfjs');

class AI{
	compile(){
		const model = tf.sequential();

		//input layer
		model.add(tf.layers.dense({
			units: 3,
			inputShape: [3]
		}))

		model.add(tf.layers.dense({
			units: 2
		}))

		model.compile({
			loss: 'meanSquaredError',
			optimizer: 'sgd'
		})

		return model;
	}

	run(){
		const model = this.compile();
		const input = tf.tensor2d([
				[0.5, 0.3, 0.4],
				[0.3, 0.4, 0.5],
				[0.4, 0.2, 0.1]				
			]);
		const output = tf.tensor2d([
				[1, 0],
				[0, 1],
				[1, 1]
			]);
		model.fit(input, output, {epochs: 10000}).then(() => {
			const data = tf.tensor2d([
				[0.3, 0.4, 0.5]
			])
			const prediction = model.predict(data);
			prediction.print();
		});
	}
}

const ai = new AI();
ai.run();