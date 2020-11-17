<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Edit Dataset</div>
      </v-card-title>
      <v-card-text>
        <DatasetForm
          :dataset="datasetForm"
          title="Update Dataset"
        ></DatasetForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit">Save</v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Dataset, DatasetUpdate, DatasetCreate, DatasetFileType } from '@/api';
import { defaultDataset } from '@/interfaces';
import DatasetForm from '@/components/DatasetForm.vue';
import {
  dispatchGetDatasets,
  dispatchUpdateDataset,
} from '@/store/dataset/actions';
import { component } from 'vue/types/umd';
import { readOneDataset } from '@/store/dataset/getters';
import { filterUndefined, deepCopy } from '@/utils';

@Component({ components: { DatasetForm } })
export default class EditDataset extends Vue {
  public datasetForm: DatasetUpdate = deepCopy(this.dataset);
  public valid = false;

  public async mounted() {
    await dispatchGetDatasets(this.$store);
    this.reset();
  }

  public reset() {
    this.datasetForm = deepCopy(this.dataset);
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredDataset: DatasetUpdate = filterUndefined(this.datasetForm);
      await dispatchUpdateDataset(this.$store, {
        id: this.dataset.id,
        dataset: filteredDataset,
      });
      this.$router.push('/main/datasets');
    }
  }

  get dataset() {
    return readOneDataset(this.$store)(+this.$router.currentRoute.params.id);
  }
}
</script>
